'''
 Text + All Images --> Text Classification

 tmux: til
'''

import random,os,warnings,gc,pickle,sys,re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from transformers import XLNetTokenizer, XLNetModel, TransfoXLModel
from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15
import numpy as np
# from Visual_embeddings import create_vis_embs
from helper import *
from sklearn.metrics import f1_score


# os.environ["CUDA_VISIBLE_DEVICES"]="6"


set_random_seed(42)

warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:4")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# class EncodingFramework(nn.Module):
#     def __init__(self, model_name='xlnet-large-cased'):
#         super(EncodingFramework, self).__init__()
#         self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
#         self.model = XLNetModel.from_pretrained(model_name)
# class EncodingFramework(nn.Module):
#     def __init__(self, model_name='bert-large-cased'):
#         super(EncodingFramework, self).__init__()
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)

class EncodingFramework(nn.Module):
    def __init__(self, model_name='neuml/pubmedbert-base-embeddings'):
        super(EncodingFramework, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.chunk_size = 512
        self.projection_layer = nn.Linear(768, embedding_dim)  # Projection layer to 1024
    def forward(self, text):
        # Tokenize the input while keeping special tokens intact
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[CSEP]', '[SEP]', '[CLS]']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        tokens = self.tokenizer(text, add_special_tokens=True, return_tensors="pt")
        # tokens = self.tokenizer(text, add_special_tokens=True, return_tensors="pt",max_length=2200,truncation=True,padding='max_length')
        token_cls = self.tokenizer("[CLS]", add_special_tokens = True, return_tensors = "pt")
        # Get tokenized IDs
        # input_ids = tokens['input_ids']
        input_ids = tokens['input_ids'].squeeze(0)
        # Divide input_ids into chunks of size `chunk_size`
        input_chunks = input_ids.split(self.chunk_size)
        token_embeddings_list = []
        cls_embeddings_list = []
        for chunk in input_chunks:
            chunk = chunk.unsqueeze(0).to(DEVICE)  # Add batch dimension for model processing
            outputs = self.model(input_ids=chunk)
            # Collect CLS and token embeddings for each chunk
            token_embeddings_list.append(outputs.last_hidden_state)
            cls_embeddings_list.append(outputs.last_hidden_state[:, 0, :])
        
        # Concatenate chunk embeddings
        token_embeddings = torch.cat(token_embeddings_list, dim=1)  # Combine along sequence dimension
        cls_embeddings = torch.cat(cls_embeddings_list, dim=0)  # Combine CLS embeddings from each chunk

        input_ids_cls = token_cls['input_ids']

        # input_ids = input_ids.to(DEVICE)
        input_ids_cls = input_ids_cls.to(DEVICE)

        # outputs = self.model(input_ids=input_ids)
        output_cls = self.model(input_ids=input_ids_cls)

        # Get the token embeddings (output hidden states)
        # token_embeddings = outputs.last_hidden_state.detach().cpu()
        cls_embeddings = output_cls.last_hidden_state.detach().cpu()
        
        token_embeddings = self.projection_layer(token_embeddings.to(DEVICE))
        cls_embeddings = self.projection_layer(cls_embeddings.to(DEVICE))

        total_tokens = input_ids.unsqueeze(0).size(1)

        # Define separator tokens
        separators = ["[SEP]", "[CSEP]"]
        separator_ids = self.tokenizer.convert_tokens_to_ids(separators)
        # print(separator_ids)

        # Find positions of each separator in the tokenized input
        separator_positions = {sep: [] for sep in separators}
        for sep, token_id in zip(separators, separator_ids):
            pos = (input_ids == token_id).nonzero(as_tuple=True)[0].tolist()
            separator_positions[sep] = pos

        # Split the paragraph by the [CSEP] token to get dynamic contents
        all_separated = text.split("[SEP]")
        all_separated = [content.strip() for content in all_separated if content.strip()]  # Remove any extra spaces

        no_context = all_separated[:-1]

        only_context = all_separated[-1]
        only_context_items = only_context.split("[CSEP]")
        only_context_items = [content.strip() for content in only_context_items if content.strip()]  # Remove any extra spaces
        # only_context_items[-1] = only_context_items[-1].split("[CLS]")[0].strip()
        contents = no_context + only_context_items

        # Tokenize each content part individually (for dynamic content)
        content_ids = []
        content_positions = []
        for content in contents:
            content_tokens = self.tokenizer(content, add_special_tokens=False, return_tensors="pt")['input_ids']
            content_ids.append(content_tokens)
            positions = []
            for token_id in content_tokens[0]:
                pos = (input_ids == token_id.item()).nonzero(as_tuple=True)[0].tolist()
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
    def __init__(self, texts, images, labels=None): #change
        self.texts = texts
        self.labels = labels
        self.images = images #change

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels is not None:
            label = self.labels[idx]
            img = self.images[idx]
            return text, img, label
        
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
        # Process each encoding in chunks
        for i in range(0, total_len, chunk_size):
            # Extract chunk of input
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
            # print(sep)
            # print(positions)
            embeddings = [transformer_xl_embeddings[0, pos, :] for pos in positions] if positions else []
            separator_embeddings_xl[sep] = embeddings

        # Extract the embeddings for each content part from Transformer-XL output
        content_embeddings_xl = []
        for positions in content_positions:
            content_part_embeddings = [transformer_xl_embeddings[0, pos, :] for pos in positions]
            content_embeddings_xl.append(content_part_embeddings)

        
        return transformer_xl_embeddings, separator_embeddings_xl, content_embeddings_xl

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
        pooled_output = torch.mean(sentence_vectors, dim=1)  # Shape: (num_sentences, embedding_dim)
        
        # Pass the pooled output through the feedforward layers
        x = F.relu(self.fc1(pooled_output))  # Apply ReLU activation function
        x = self.fc2(x)  # Output layer

        return torch.sigmoid(x)  # Use sigmoid for binary classification output


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    c, num_excl = 0, 0
    for texts, imgs, labels in tqdm(train_loader):

        optimizer.zero_grad()
        # Forward pass
        encodings = [framework.forward(text) for text in texts]

        xlnet_encodings, total_tokens, cls_encodings, sep, sep_pos, sep_emb, cont, cont_pos = zip(*encodings)
        
        sep_token_emb = [elem['[SEP]'][0].unsqueeze(0).unsqueeze(0) for elem in sep_emb]
        sep_all = [elem + ['[CLS]'] for elem in sep]
        for i in range(len(sep_pos)):
            sep_pos[i]['[CLS]'] = [total_tokens[i]]

        combined_encodings = [combine_text_img_token(xlnet_encodings[i], imgs[i], cls_encodings[i], sep_token_emb[i], DEVICE) for i in range(len(xlnet_encodings))]

        transformed_encodings = [transformer_framework.forward(combined_encodings[i], sep_all[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(combined_encodings))]

        transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs = [list(group) for group in list(zip(*transformed_encodings))]
        transfoxl_context_sentences = []
        for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
            context = elem[3:]
            transfoxl_context_sentences.append(context)
        attn_context_sentences = [apply_self_attention(context, self_attention_layer, DEVICE) for context in transfoxl_context_sentences]
        fl_sentence_vectors, padded_masks = [list(group) for group in list(zip(*attn_context_sentences))]
        fl_sentence_vectors_cls= attended_sentence_cls(fl_sentence_vectors, transfoxl_sep_embs, padded_masks, DEVICE)
        if len(fl_sentence_vectors_cls) == batch_size:
            # inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
            #                         for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks_cls)]
            inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                    for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks)]
            attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]

            predictions = [classification_head(vectors) for vectors in attended_sentence_output]
            logits = convert_tensor([(pred> 0.5).float() for pred in predictions])
            padded_logits, padded_labels = pad_tensor_lists(logits, labels.float())
            padded_labels = padded_labels.to(DEVICE)
            padded_logits = padded_logits.to(DEVICE)

            loss = criterion(padded_logits, padded_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # wandb.log({"batch_loss": loss.item()})
        else:
            num_excl = num_excl + 1
            print("Batch {} excluded".format(c))
            with open("bath_excluded.txt", 'a') as f:
                f.write(f"Batch: {c} excluded from training %\n")
        
        # wandb.log({"epoch_train_loss": train_loss})
        c = c + 1
    return total_loss / (len(train_loader) - num_excl)

def evaluate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    accuracy_sample, f1_score_sample = 0, 0
    c, num_excl = 0, 0
    with torch.no_grad():
        for texts, imgs, labels in tqdm(valid_loader):

            encodings = [framework.forward(text) for text in texts]
            xlnet_encodings, total_tokens, cls_encodings, sep, sep_pos, sep_emb, cont, cont_pos = [list(group) for group in list(zip(*encodings))]
            sep_token_emb = [elem['[SEP]'][0].unsqueeze(0).unsqueeze(0) for elem in sep_emb]
            sep_all = [elem + ['[CLS]'] for elem in sep]
            for i in range(len(sep_pos)):
                sep_pos[i]['[CLS]'] = [total_tokens[i]]
            combined_encodings = [combine_text_img_token(xlnet_encodings[i], imgs[i], cls_encodings[i], sep_token_emb[i], DEVICE) for i in range(len(xlnet_encodings))]

            transformed_encodings = [transformer_framework.forward(combined_encodings[i], sep_all[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(combined_encodings))]


            transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs = [list(group) for group in list(zip(*transformed_encodings))]
            transfoxl_context_sentences = []
            for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
                context = elem[3:]
                transfoxl_context_sentences.append(context)
            attn_context_sentences = [apply_self_attention(context, self_attention_layer, DEVICE) for context in transfoxl_context_sentences]
            fl_sentence_vectors, padded_masks = [list(group) for group in list(zip(*attn_context_sentences))]
            fl_sentence_vectors_cls= attended_sentence_cls(fl_sentence_vectors, transfoxl_sep_embs, padded_masks, DEVICE)

            inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks)]
    
            attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]
    
            predictions = [classification_head(vectors) for vectors in attended_sentence_output]
            logits = convert_tensor([(pred> 0.5).float() for pred in predictions])
            padded_logits, padded_labels = pad_tensor_lists(logits, labels.float())
            padded_labels = padded_labels.to(DEVICE)
            padded_logits = padded_logits.to(DEVICE)
            loss = criterion(padded_logits, padded_labels)

            total_loss += loss.item()
            accuracy_sample += calc_accuracy(padded_logits, padded_labels)
            f1_score_sample = f1_score_sample + f1_score(padded_labels.cpu(), padded_logits.cpu(), average='macro')  # Use 'micro' or 'weighted' as needed

    
    c = c + 1
    # wandb.log({"epoch_valid_loss": valid_loss, "epoch_valid_accuracy": valid_accuracy})
    return total_loss / (len(valid_loader) - num_excl), accuracy_sample / (len(valid_loader) - num_excl), f1_score_sample / (len(valid_loader) - num_excl)

def test_model(model, test_loader):
    model.eval()
    accuracy_sample, f1_score_sample = 0, 0
    c, num_excl = 0, 0
    with torch.no_grad():
        for texts, imgs, labels in tqdm(test_loader):

            encodings = [framework.forward(text) for text in texts]
            xlnet_encodings, total_tokens, cls_encodings, sep, sep_pos, sep_emb, cont, cont_pos = [list(group) for group in list(zip(*encodings))]
            sep_token_emb = [elem['[SEP]'][0].unsqueeze(0).unsqueeze(0) for elem in sep_emb]
            sep_all = [elem + ['[CLS]'] for elem in sep]
            for i in range(len(sep_pos)):
                sep_pos[i]['[CLS]'] = [total_tokens[i]]
            combined_encodings = [combine_text_img_token(xlnet_encodings[i], imgs[i], cls_encodings[i], sep_token_emb[i], DEVICE) for i in range(len(xlnet_encodings))]

            transformed_encodings = [transformer_framework.forward(combined_encodings[i], sep_all[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(combined_encodings))]


            transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs = [list(group) for group in list(zip(*transformed_encodings))]
            transfoxl_context_sentences = []
            for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
                context = elem[3:]
                transfoxl_context_sentences.append(context)
            attn_context_sentences = [apply_self_attention(context, self_attention_layer, DEVICE) for context in transfoxl_context_sentences]
            fl_sentence_vectors, padded_masks = [list(group) for group in list(zip(*attn_context_sentences))]
            fl_sentence_vectors_cls= attended_sentence_cls(fl_sentence_vectors, transfoxl_sep_embs, padded_masks, DEVICE)

            inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks)]
    
            attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]
    
            predictions = [classification_head(vectors) for vectors in attended_sentence_output]
            logits = convert_tensor([(pred> 0.5).float() for pred in predictions])
            padded_logits, padded_labels = pad_tensor_lists(logits, labels.float())
            padded_labels = padded_labels.to(DEVICE)
            padded_logits = padded_logits.to(DEVICE)

            accuracy_sample += calc_accuracy(padded_logits, padded_labels)
            f1_score_sample = f1_score_sample + f1_score(padded_labels.cpu(), padded_logits.cpu(), average='macro')  # Use 'micro' or 'weighted' as needed

    
    c = c + 1
    # wandb.log({"epoch_valid_loss": valid_loss, "epoch_valid_accuracy": valid_accuracy})
    return accuracy_sample / (len(test_loader) - num_excl), f1_score_sample / (len(test_loader) - num_excl)

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
# img_emb = create_vis_embs("Dataset/question_image_dict.csv")
    
for key, value in img_emb.items():
    if len(value) > 1:
        img_emb[key] = torch.mean(torch.stack(value), dim=0)
    # elif len(value) == 0:
    #     img_emb[key] = [torch.ones((1, 768))] #white_image_tensor
# --------------------------------- STEP 1 -------------------------------------------
''' Create encodings by passing the text through XLNet. Record the positions of the
    seperaters and each element of the input, i.e, question, intent, type and each
    individual sentence of context separately
'''

embedding_dim = 1024 # This is the embeddings dimension of each of transfoxl_embs, can be obtained otherwise by: transfoxl_embs[i].size(-1) 
hidden_size = 512
image_size = 1024

num = len(list(QID_context.values()))
# num = 20
labels = []
corr_context = list(QID_context.values())[:num]
corr_ans = list(QID_ans.values())[:num]
# corr_context = list(QID_context.values())
# corr_ans = list(QID_ans.values())
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

# print(labels)
combined_texts = list(QID_q_int_type_cont.values())[:num]
images = list(new_image_dict.values())[:num]
 
conv_images = []
for img in tqdm(images):
    if len(img) > 0:
        conv_images.append(convert_img_shape(img[0].to(DEVICE), image_size, embedding_dim, DEVICE))
    else:
        # print(img)
        conv_images.append(torch.tensor(img))


train_texts, temp_texts, train_img, temp_img, train_labels, temp_labels = train_test_split(combined_texts,conv_images, labels, test_size=0.3, random_state=42)
valid_texts, test_texts, valid_img, test_img, valid_labels, test_labels = train_test_split(temp_texts,temp_img, temp_labels, test_size=0.5, random_state=42)

print(f"Training set: {len(train_texts)} examples")
print(f"Validation set: {len(valid_texts)} examples")
print(f"Test set: {len(test_texts)} examples")

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_img, train_labels)
valid_dataset = CustomDataset(valid_texts, valid_img, valid_labels)
test_dataset = CustomDataset(test_texts, test_img, test_labels)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_3, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn_3, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_3, shuffle=False)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_3, shuffle=True, num_workers=4, pin_memory=True)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn_3, shuffle=False, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_3, shuffle=False, num_workers=4, pin_memory=True)


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
# model = torch.nn.DataParallel(model)  # Wrap your model for multi-GPU use
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

# wandb.login()
# wandb.init(project='MEDQA-IMG',)
# wandb.config = {
#   "learning_rate": 1e-5,
#   "epochs": 10,
#   "batch_size": 4
# }
import os
# Training loop
file_path = "tip/output.txt"
checkpoint_path = "tip/tip.pth"
# Ensure directories exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)
EPOCHS = 25
for epoch in tqdm(range(EPOCHS)):  # Number of epochs
    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_accuracy, valid_f1 = evaluate_model(model, valid_loader, criterion)

    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {valid_loss:.4f}')
    print(f'Validation Accuracy: {valid_accuracy:.4f}')
    
    checkpoint_path = "til/til.pth"
    torch.save({'epoch': epoch,                        # Current epoch
    'model_state_dict': model.state_dict(), # Model parameters
    'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

    test_accuracy, test_f1 = test_model(model, test_loader)
    test_accuracy = test_accuracy * 100
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1: {test_f1:.4f}')


    log_epoch_info_3(file_path, epoch, train_loss, valid_loss, valid_accuracy, valid_f1, test_accuracy, test_f1)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test F1: {test_f1:.4f}')

# wandb.finish()