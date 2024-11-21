 
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from transformers import XLNetTokenizer, XLNetModel, TransfoXLModel
from transformers import LongformerTokenizer, LongformerModel
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import re
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15
import os
import numpy as np
import warnings
import gc
import pickle
import sys #,wandb

from Visual_embeddings import create_vis_embs
# from helper import set_random_seed, convert_context, create_labels, collate_fn
# from helper import mean_pooling, apply_self_attention, convert_tensor, pad_tensor_lists
# from helper import calc_accuracy, log_epoch_info, convert_img_shape, combine_text_img_token
# from helper import attended_sentence_cls, concat_images, attended_cls_icls

from helper import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="3,4,7"


set_random_seed(42)



warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:5")
    print("Using GPU", DEVICE)
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")



class EncodingFramework(nn.Module):
    def __init__(self, model_name='allenai/longformer-base-4096'):
        super(EncodingFramework, self).__init__()
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.projection_layer = nn.Linear(768, embedding_dim)  # Projection layer to 1024

    def forward(self, text):
        # Tokenize the input while keeping special tokens intact
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[CSEP]', '[SEP]', '[CLS]']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        tokens = self.tokenizer(text, add_special_tokens=True, return_tensors="pt",max_length=2200,truncation=True,padding='max_length')
        token_cls = self.tokenizer("[CLS]", return_tensors = "pt")

        # Get tokenized IDs
        input_ids = tokens['input_ids']
        input_ids_cls = token_cls['input_ids']

        input_ids = input_ids.to(DEVICE)
        input_ids_cls = input_ids_cls.to(DEVICE)

        outputs = self.model(input_ids=input_ids)
        output_cls = self.model(input_ids=input_ids_cls)

        # Get the token embeddings (output hidden states)
        token_embeddings = outputs.last_hidden_state.detach().cpu()
        cls_embeddings = output_cls.last_hidden_state.detach().cpu()
        token_embeddings = self.projection_layer(token_embeddings.to(DEVICE))
        cls_embeddings = self.projection_layer(cls_embeddings.to(DEVICE))
        total_tokens = input_ids.size(1)

        # Define separator tokens
        separators = ["[SEP]", "[CSEP]"]
        separator_ids = self.tokenizer.convert_tokens_to_ids(separators)

        # Find positions of each separator in the tokenized input
        separator_positions = {sep: [] for sep in separators}
        for sep, token_id in zip(separators, separator_ids):
            pos = (input_ids == token_id).nonzero(as_tuple=True)[1].tolist()
            separator_positions[sep] = pos

        # Split the paragraph by the [CSEP] token to get dynamic contents
        all_separated = text.split("[SEP]")
        all_separated = [content.strip() for content in all_separated if content.strip()]  # Remove any extra spaces

        no_context = all_separated[:-1]

        only_context = all_separated[-1]
        only_context_items = only_context.split("[CSEP]")
        only_context_items = [content.strip() for content in only_context_items if content.strip()]  # Remove any extra spaces
        contents = no_context + only_context_items

        # Tokenize each content part individually (for dynamic content)
        content_ids = []
        content_positions = []
        for content in contents:
            content_tokens = self.tokenizer(content, add_special_tokens=False, return_tensors="pt")['input_ids']
            content_ids.append(content_tokens)
            positions = []
            for token_id in content_tokens[0]:
                pos = (input_ids == token_id.item()).nonzero(as_tuple=True)[1].tolist()
                if pos:
                    positions.append(pos[0])
            content_positions.append(positions)
        separator_embeddings = {}
        for sep in separators:
            positions = separator_positions[sep]
            embeddings = [token_embeddings[0, pos, :].detach().cpu() for pos in positions] if positions else []
            separator_embeddings[sep] = embeddings

        return token_embeddings, total_tokens, cls_embeddings, separators, separator_positions, separator_embeddings, contents, content_positions

# Define the dataset
class CustomDataset(Dataset):
    def __init__(self, texts, images, labels=None, img_labels = None): #change
        self.texts = texts
        self.labels = labels
        self.images = images #change
        self.img_labels = img_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels is not None and self.img_labels is not None:
            label = self.labels[idx]
            img = self.images[idx]
            img_label = self.img_labels[idx]
            return text, img, label, img_label
        
        return text


class TransformerXLFramework(nn.Module):
    def __init__(self, model_name="transfo-xl/transfo-xl-wt103"):
        super(TransformerXLFramework, self).__init__()
        self.model = TransfoXLModel.from_pretrained(model_name)


    def forward(self, encodings, separators, separator_positions, contents, content_positions):


        total_len = encodings.size(1)
        transformer_xl_embeddings = []
        memory = None  # Initialize memory to None for the first segment
        chunk_size = 512
        for i in range(0, total_len, chunk_size):
            chunk_encodings = encodings[:, i:i + chunk_size].to(DEVICE)  # Move chunk to GPU

            # Pass chunk through Transformer-XL, along with memory from the previous chunk
            transformer_xl_output = self.model(inputs_embeds=chunk_encodings, mems=memory)

            # Get the token embeddings from Transformer-XL output
            transformer_xl_embeddings.append(transformer_xl_output.last_hidden_state.detach().cpu())  # Move output to CPU to save GPU memory

            # Update memory with the current chunk's memory, move memory to GPU for next iteration
            memory = [mem.to(DEVICE) for mem in transformer_xl_output.mems]


        # Concatenate all chunk embeddings along the sequence dimension
        transformer_xl_embeddings = torch.cat(transformer_xl_embeddings, dim=1)

        # Extract the embeddings for each separator token from Transformer-XL output
        separator_embeddings_xl = {}
        for sep in separators:
            positions = separator_positions[sep]
            embeddings = [transformer_xl_embeddings[0, pos, :] for pos in positions] if positions else []
            separator_embeddings_xl[sep] = embeddings

        img_sep_pos = [separator_positions['[CLS]'][0] + 1]
        img_sep_emb = [transformer_xl_embeddings[0, pos, :] for pos in img_sep_pos] if img_sep_pos else []
        separator_embeddings_xl['[SEP]'].append(img_sep_emb)

        icls_pos = [transformer_xl_embeddings.shape[1] - 1]
        icls_emb = [transformer_xl_embeddings[0, pos, :] for pos in icls_pos] if icls_pos else []
        separator_embeddings_xl['[ICLS]'] = icls_emb
        separator_positions['[ICLS]'] = icls_pos

        cls_emb = [transformer_xl_embeddings[0, pos, :] for pos in separator_positions['[CLS]']] if separator_positions['[CLS]'] else []

        i = 0
        img_embs, isep_embs, isep_pos, img_pos = [],[], [], []
        img_enc, isep_enc = [], []
        for pos in range(img_sep_pos[0] + 1, icls_pos[0]):
            emb = [transformer_xl_embeddings[0, pos, :]]
            enc = [encodings[0, pos, :]]
            if i % 2 == 0:
                img_embs.append(emb)
                img_pos.append(pos)
                img_enc.append(enc)
            else:
                isep_embs.append(emb)
                isep_pos.append(pos)
                isep_enc.append(enc)
            i = i + 1

        separators = separators + ['[ISEP]', '[ICLS]']
        separator_embeddings_xl['[ISEP]'] = isep_embs
        separator_positions['[ISEP]'] = isep_pos

        

        # Extract the embeddings for each content part from Transformer-XL output
        content_embeddings_xl = []
        for positions in content_positions:
            content_part_embeddings = [transformer_xl_embeddings[0, pos, :] for pos in positions]
            content_embeddings_xl.append(content_part_embeddings)

        return transformer_xl_embeddings, separator_embeddings_xl, content_embeddings_xl, img_embs
    

class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttentionLayer, self).__init__()
        # Define the necessary layers for self-attention
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Self-attention computation
        Q = self.query(x)   # Query
        K = self.key(x)     # Key
        V = self.value(x)   # Value
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention weights to the values
        attended_output = torch.matmul(attention_weights, V)

        return attended_output
    

class InterSentenceSelfAttention(nn.Module):
    def __init__(self, embedding_dim, alpha=1.5):
        super(InterSentenceSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.alpha = alpha  # α parameter for α-entmax (e.g., 1.5 or 1.3)
        
        # Linear layers for attention mechanism
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, sentence_embeddings, mask=None):
        # Compute Query, Key, and Value matrices
        if sentence_embeddings.numel() == 0:
            return torch.tensor([]), torch.tensor([])
        else:
            Q = self.query(sentence_embeddings)  # Shape: (num_sentences, max_seq_len, embedding_dim)
            K = self.key(sentence_embeddings)    # Shape: (num_sentences, max_seq_len, embedding_dim)
            V = self.value(sentence_embeddings)  # Shape: (num_sentences, max_seq_len, embedding_dim)
            
            # Compute attention scores (inter-sentence self-attention)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # Shape: (num_sentences, max_seq_len, max_seq_len)

            # Apply masking (to ignore padding tokens in the attention)
            if mask is not None:
                # Ensure the mask is expanded correctly for broadcasting
                mask_expanded = mask.unsqueeze(1).expand(-1, attention_scores.size(1), -1)  # Shape: (num_sentences, 1, max_seq_len)
                mask_expanded = mask_expanded.to(DEVICE)
                attention_scores = attention_scores.masked_fill(mask_expanded == 0, float('-inf'))
            
            # Apply α-entmax for sparse attention weights
            attention_weights = entmax15(attention_scores, dim=-1)  # Shape: (num_sentences, max_seq_len, max_seq_len)

            # Apply the sparse attention weights to the value matrix (V)
            attended_output = torch.matmul(attention_weights, V)  # Shape: (num_sentences, max_seq_len, embedding_dim)

            return attended_output, attention_weights
    
class SentenceRelevanceIdentifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SentenceRelevanceIdentifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # First layer
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer for binary classification

    def forward(self, sentence_vectors):
        # Apply mean pooling over the sequence length
        if sentence_vectors.numel() == 0:
            return torch.tensor([])
        else:
            pooled_output = torch.mean(sentence_vectors, dim=1)  # Shape: (num_sentences, embedding_dim)
            
            # Pass the pooled output through the feedforward layers
            x = F.relu(self.fc1(pooled_output))  # Apply ReLU activation function
            x = self.fc2(x)  # Output layer

            return torch.sigmoid(x)  # Use sigmoid for binary classification output


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    c, num_excl = 0, 0
    for texts, imgs, labels, img_labels in tqdm(train_loader):
        optimizer.zero_grad()
        encodings = [framework.forward(text) for text in texts]
        xlnet_encodings, total_tokens, cls_encodings,  sep, sep_pos, sep_emb, cont, cont_pos = zip(*encodings)
        
        sep_token_emb = [elem['[SEP]'][0].unsqueeze(0).unsqueeze(0) for elem in sep_emb]
        sep_all = [elem + ['[CLS]'] for elem in sep]
        for i in range(len(sep_pos)):
            sep_pos[i]['[CLS]'] = [total_tokens[i]]
        cls_encodings = [cl_enc.mean(dim=1, keepdim=True) for cl_enc in cls_encodings]

        combined_encodings = [combine_text_img_token(xlnet_encodings[i], imgs[i], cls_encodings[i], sep_token_emb[i], DEVICE) for i in range(len(xlnet_encodings))]

        transformed_encodings = [transformer_framework.forward(combined_encodings[i], sep_all[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(combined_encodings))]

        transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs, img_embs = [list(group) for group in list(zip(*transformed_encodings))]
        transfoxl_context_sentences = []
        for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
            context = elem[3:]
            transfoxl_context_sentences.append(context)
        # breakpoint()
        attn_context_sentences = [apply_self_attention(context, self_attention_layer, DEVICE) for context in transfoxl_context_sentences]
        attn_imgs = [apply_self_attention(img, self_attention_layer, DEVICE) for img in img_embs]
        
        fl_sentence_vectors, padded_masks_sent = [list(group) for group in list(zip(*attn_context_sentences))]
        fl_img_vectors, padded_masks_img = [list(group) for group in list(zip(*attn_imgs))]

        fl_sentence_vectors_cls= [attended_cls_icls(fl_sentence_vectors[i], transfoxl_sep_embs[i], padded_masks_sent[i], DEVICE) for i in range(len(fl_sentence_vectors))]
        
        fl_img_vectors_cls= [attended_cls_icls(fl_img_vectors[i], transfoxl_sep_embs[i], padded_masks_img[i], DEVICE) for i in range(len(fl_img_vectors))]

        padded_masks_sent = [torch.cat((mask, torch.ones((mask.shape[0], 2), dtype=mask.dtype)), dim=1) for mask in padded_masks_sent]
        padded_masks_img = [torch.cat((mask, torch.ones((mask.shape[0], 2), dtype=mask.dtype)), dim=1) for mask in padded_masks_img]
        if len(fl_sentence_vectors_cls) == batch_size and len(fl_img_vectors_cls) ==batch_size:
            inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                    for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks_sent)]
            attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]

            inter_attended_imgs = [inter_sentence_attention_layer(final_img_embedding, padded_mask)
                                    for (final_img_embedding, padded_mask) in zip(fl_img_vectors_cls, padded_masks_img)]
            attended_img_output, inter_img_attention_weights = [list(group) for group in list(zip(*inter_attended_imgs))]


            predictions_sent = [classification_head(vectors) for vectors in attended_sentence_output]
            predictions_img = [classification_head(vectors) for vectors in attended_img_output]

            logits_sent = convert_tensor([(pred> 0.5).float() for pred in predictions_sent])
            logits_img = convert_tensor([(pred> 0.5).float() for pred in predictions_img])
            
            padded_logits_sent, padded_labels_sent = pad_tensor_lists(logits_sent, labels.float())
            padded_labels_sent = padded_labels_sent.to(DEVICE)
            padded_logits_sent = padded_logits_sent.to(DEVICE)

            logits_img = logits_img.to(DEVICE)
            labels_img = img_labels.float().to(DEVICE)

            padded_logits_img, padded_labels_img = pad_tensor_lists(logits_img, labels_img.float())
            padded_labels_img = padded_labels_img.to(DEVICE)
            padded_logits_img = padded_logits_img.to(DEVICE)
            
            loss_sent = criterion(padded_logits_sent, padded_labels_sent)
            loss_img = criterion(padded_logits_img, padded_labels_img)
            loss = 0.7 * loss_sent + 0.3 * loss_img
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        else:
            num_excl = num_excl + 1
            print("Batch {} excluded".format(c))
            with open("bath_excluded.txt", 'a') as f:
                f.write(f"Batch: {c} excluded from training %\n")

        c = c + 1
    return total_loss / (len(train_loader) - num_excl)

def evaluate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    accuracy_sent, accuracy_img, f1_score_sample = 0, 0, 0
    c, num_excl = 0, 0
    with torch.no_grad():
        for texts, imgs, labels, img_labels in tqdm(valid_loader):

            encodings = [framework.forward(text) for text in texts]
            xlnet_encodings, total_tokens, cls_encodings, sep, sep_pos, sep_emb, cont, cont_pos = [list(group) for group in list(zip(*encodings))]
            sep_token_emb = [elem['[SEP]'][0].unsqueeze(0).unsqueeze(0) for elem in sep_emb]
            sep_all = [elem + ['[CLS]'] for elem in sep]
            for i in range(len(sep_pos)):
                sep_pos[i]['[CLS]'] = [total_tokens[i]]
            combined_encodings = [combine_text_img_token(xlnet_encodings[i], imgs[i], cls_encodings[i], sep_token_emb[i], DEVICE) for i in range(len(xlnet_encodings))]

            transformed_encodings = [transformer_framework.forward(combined_encodings[i], sep_all[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(combined_encodings))]


            transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs, img_embs = [list(group) for group in list(zip(*transformed_encodings))]
            transfoxl_context_sentences = []
            for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
                context = elem[3:]
                transfoxl_context_sentences.append(context)
            # breakpoint()
            attn_context_sentences = [apply_self_attention(context, self_attention_layer, DEVICE) for context in transfoxl_context_sentences]
            attn_imgs = [apply_self_attention(img, self_attention_layer, DEVICE) for img in img_embs]
            
            fl_sentence_vectors, padded_masks_sent = [list(group) for group in list(zip(*attn_context_sentences))]
            fl_img_vectors, padded_masks_img = [list(group) for group in list(zip(*attn_imgs))]

            fl_sentence_vectors_cls= [attended_cls_icls(fl_sentence_vectors[i], transfoxl_sep_embs[i], padded_masks_sent[i], DEVICE) for i in range(len(fl_sentence_vectors))]
            fl_img_vectors_cls= [attended_cls_icls(fl_img_vectors[i], transfoxl_sep_embs[i], padded_masks_img[i], DEVICE) for i in range(len(fl_img_vectors))]

            padded_masks_sent = [torch.cat((mask, torch.ones((mask.shape[0], 2), dtype=mask.dtype)), dim=1) for mask in padded_masks_sent]
            padded_masks_img = [torch.cat((mask, torch.ones((mask.shape[0], 2), dtype=mask.dtype)), dim=1) for mask in padded_masks_img]

            inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                    for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks_sent)]
            attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]

            inter_attended_imgs = [inter_sentence_attention_layer(final_img_embedding, padded_mask)
                                    for (final_img_embedding, padded_mask) in zip(fl_img_vectors_cls, padded_masks_img)]
            attended_img_output, inter_img_attention_weights = [list(group) for group in list(zip(*inter_attended_imgs))]


            predictions_sent = [classification_head(vectors) for vectors in attended_sentence_output]
            predictions_img = [classification_head(vectors) for vectors in attended_img_output]

            logits_sent = convert_tensor([(pred> 0.5).float() for pred in predictions_sent])
            logits_img = convert_tensor([(pred> 0.5).float() for pred in predictions_img])
            
            padded_logits_sent, padded_labels_sent = pad_tensor_lists(logits_sent, labels.float())
            padded_labels_sent = padded_labels_sent.to(DEVICE)
            padded_logits_sent = padded_logits_sent.to(DEVICE)



            logits_img = logits_img.to(DEVICE)
            labels_img = img_labels.float().to(DEVICE)
            padded_logits_img, padded_labels_img = pad_tensor_lists(logits_img, labels_img.float())
            padded_labels_img = padded_labels_img.to(DEVICE)
            padded_logits_img = padded_logits_img.to(DEVICE)
            loss_sent = criterion(padded_logits_sent, padded_labels_sent)
            loss_img = criterion(padded_logits_img, padded_labels_img)
            loss = 0.7 * loss_sent + 0.3 * loss_img
            total_loss += loss.item()

            accuracy_sent += calc_accuracy(padded_logits_sent, padded_labels_sent)
            accuracy_img += calc_accuracy(padded_logits_img, padded_labels_img)

    c = c + 1
    return total_loss / (len(valid_loader) - num_excl), accuracy_sent / (len(valid_loader) - num_excl), accuracy_img / (len(valid_loader) - num_excl)

# --------------------------------------------------------------------------------------------------------------------------------
'''
context_qa_list: 
    -type: list
    -contents: individual dictionaries of the form {[Context 1]: context,
    QID: {Que: question [SEP] intent [SEP] type}, Ans: answer}}

QID_context:
    -type: dictionary
    -contents: individual dictionaries of the form {QID: context}


QID_ans:
    -type: dictionary
    -contents: individual dictionaries of the form {QID: answer}

QID_ques:
    -type: dictionary
    -contents: individual dictionaries of the form {QID: question [SEP]
    intent [SEP] type}


QID_q_context:
    -type: dictionary
    -contents: individual dictionaries of the form {QID: question [SEP] 
    intent [SEP] type [SEP] context}
'''
QID_context = pd.read_pickle('QID_context.pkl')
QID_ans = pd.read_pickle('QID_ans.pkl')
QID_ques = pd.read_pickle('QID_ques.pkl')
QID_q_context = pd.read_pickle('QID_q_context.pkl')
context_qa_list = pd.read_pickle('context_qa_list.pkl')
QID_q_int_type_cont = pd.read_pickle('QID_q_int_type_cont.pkl')
img_emb = pd.read_pickle('full_image_embeddings_cpu.pkl')
image_labels = pd.read_pickle('image_labels.pkl')

# --------------------------------- STEP 1 -------------------------------------------
''' Create encodings by passing the text through XLNet. Record the positions of the
    seperaters and each element of the input, i.e, question, intent, type and each
    individual sentence of context separately
'''


embedding_dim = 1024 # This is the embeddings dimension of each of transfoxl_embs, can be obtained otherwise by: transfoxl_embs[i].size(-1) 
hidden_size = 512
image_size = 1024

num = len(list(QID_context.values()))
labels = []
corr_context = list(QID_context.values())[:num]
corr_ans = list(QID_ans.values())[:num]
for i in range(len(corr_context)):
  context_sent = corr_context[i].split('.')
  ans_sent = corr_ans[i].split('.')
  label = create_labels(context_sent, ans_sent)
  labels.append(label)

new_image_dict = dict()
for key, value in QID_ans.items():
    if key in list(img_emb.keys()):
        new_image_dict[key] = img_emb[key]
    else:
        new_image_dict[key] = []

new_img_labels_dict = dict()
for key, value in QID_ans.items():
    if key in list(image_labels.keys()):
        new_img_labels_dict[key] = image_labels[key]
    else:
        new_img_labels_dict[key] = []

combined_texts = list(QID_q_int_type_cont.values())[:num]
images = list(new_image_dict.values())[:num] #change
img_labels = list(new_img_labels_dict.values())[:num]

images_combined = []
for img in tqdm(images):
    images_combined.append(concat_images(img, DEVICE))

with open('images_combined_full.pkl', 'wb') as f:
    pickle.dump(images_combined, f)

# images_combined = pd.read_pickle("images_combined_1000.pkl")
# breakpoint()
 
conv_images = []
for img in tqdm(images_combined):
    if len(img) > 0:
        conv_images.append(convert_img_shape(img, image_size, embedding_dim, DEVICE))
    else:
        # print(img)
        conv_images.append(torch.tensor(img))

train_texts, temp_texts, train_img, temp_img, train_labels, temp_labels, train_img_labels, temp_img_labels = train_test_split(combined_texts,conv_images, labels, img_labels, test_size=0.3, random_state=42)
valid_texts, test_texts, valid_img, test_img, valid_labels, test_labels, valid_img_labels, test_img_labels = train_test_split(temp_texts,temp_img, temp_labels, temp_img_labels, test_size=0.5, random_state=42)

print(f"Training set: {len(train_texts)} examples")
print(f"Validation set: {len(valid_texts)} examples")
print(f"Test set: {len(test_texts)} examples")

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_img, train_labels, train_img_labels)
valid_dataset = CustomDataset(valid_texts, valid_img, valid_labels, valid_img_labels)
test_dataset = CustomDataset(test_texts, test_img, test_labels, test_img_labels)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)



# Define the model, criterion, and optimizer
framework = EncodingFramework()
transformer_framework = TransformerXLFramework()
self_attention_layer = SelfAttentionLayer(embedding_dim)
inter_sentence_attention_layer = InterSentenceSelfAttention(embedding_dim, alpha=1.5)
classification_head = SentenceRelevanceIdentifier(embedding_dim, hidden_size)
model = torch.nn.ModuleList([
    framework, 
    transformer_framework, 
    self_attention_layer, 
    inter_sentence_attention_layer, 
    classification_head
])
model = model.to(DEVICE)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam([
    {'params': framework.parameters()},
    {'params': transformer_framework.parameters()},
    {'params': self_attention_layer.parameters()},
    {'params': inter_sentence_attention_layer.parameters()},
    {'params': classification_head.parameters()},
], lr=3e-5)


# --------------------------------- STEP 2 -------------------------------------------
''' Pass the XLNet encodings through Transformer XL. Maintain the positions of the
    seperators and contents as in STEP 1

    Important Notes:
        1. Transformer excel expects the tensor to have a hidden size of 1024.
            So, we need to use xlnet-large-cased instead of xlnet-base-cased (hidden_size
            = 768)

        2. A small change needed to be made in the file for transformer-xl.
            
            Filepath: /mnt/Data/abhisek_1921cs16/anaconda3/envs/Multisum/lib/python3.10/site-packages/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py
            
            Replace at line 941:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=torch.int64).type_as(
                dtype=word_emb.dtype
            )

            With:
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=torch.int64).type_as(word_emb)
'''


# Training loop
file_path = "ticl/output.txt"
checkpoint_path = "ticl/ticl.pth"
# Ensure directories exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)
EPOCHS = 25
for epoch in tqdm(range(EPOCHS)):  # Number of epochs
    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_accuracy_sent, valid_accuracy_img = evaluate_model(model, valid_loader, criterion)

    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {valid_loss:.4f}')
    print(f'Validation Accuracy Sentences:  {valid_accuracy_sent:.4f}')
    print(f'Validation Accuracy Images:  {valid_accuracy_sent:.4f}')

    torch.save({'epoch': epoch,                        # Current epoch
    'model_state_dict': model.state_dict(), # Model parameters
    'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

    log_epoch_info(file_path, epoch, train_loss, valid_loss, valid_accuracy_sent, valid_accuracy_img)