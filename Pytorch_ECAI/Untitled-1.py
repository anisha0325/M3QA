
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from transformers import XLNetTokenizer, XLNetModel, TransfoXLModel
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15
import os
import numpy as np
import warnings
import gc
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

def set_random_seed(seed: int):
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(42)



warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Add [CSEP] tokens in between all context sentences 
def convert_context(context):
    sentences = re.split(r'\.\s*', context)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    csep_sentences = ' [SEP] ' + ' [CSEP] '.join(sentences) + ' [CLS] '
    gc.collect()
    return csep_sentences

def create_labels(context_sentences, answer_sentences):

    labels = [1 if sentence in answer_sentences else 0 for sentence in context_sentences]
    return labels

class EncodingFramework(nn.Module):
    def __init__(self, model_name='xlnet-large-cased'):
        super(EncodingFramework, self).__init__()
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetModel.from_pretrained(model_name)

    def forward(self, text):
        # Tokenize the input while keeping special tokens intact
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[CSEP]', '[SEP]', '[CLS]']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        tokens = self.tokenizer(text, add_special_tokens=True, return_tensors="pt")

        # Get tokenized IDs
        input_ids = tokens['input_ids']
        # print(f"Token IDs: {input_ids}")

        # Define separator tokens
        separators = ["[SEP]", "[CSEP]", "[CLS]"]
        separator_ids = self.tokenizer.convert_tokens_to_ids(separators)
        # print(separator_ids)

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
        only_context_items[-1] = only_context_items[-1].split("[CLS]")[0].strip()
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
            # print(positions)

        # Pass the tokenized input through the XLNet model to get the embeddings
        input_ids = input_ids.to(DEVICE)
        outputs = self.model(input_ids=input_ids)

        # Get the token embeddings (output hidden states)
        token_embeddings = outputs.last_hidden_state

        del input_ids
        del outputs
        del tokens
        del all_separated
        del no_context
        del only_context
        del only_context_items
        del content_ids

        gc.collect()
        torch.cuda.empty_cache()
        return token_embeddings, separators, separator_positions, contents, content_positions

# Define the dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return text, label
        gc.collect()
        return text
    
# Define collate function
def collate_fn(batch):
    texts, labels = zip(*batch)
    # texts_padded = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Determine the maximum length of labels in the batch
    max_label_length = max(len(label) for label in labels)

    # Pad labels
    labels_padded = torch.full((len(labels), max_label_length), fill_value=0, dtype=torch.long)  # Using -1 as padding value
    for i, label in enumerate(labels):
        labels_padded[i, :len(label)] = torch.tensor(label, dtype=torch.long)
    gc.collect()
    return texts, labels_padded


class TransformerXLFramework(nn.Module):
    def __init__(self, model_name="transfo-xl/transfo-xl-wt103"):
        super(TransformerXLFramework, self).__init__()
        self.model = TransfoXLModel.from_pretrained(model_name)


    def forward(self, encodings, separators, separator_positions, contents, content_positions):

        # Transformer-XL expects input in 2D format, so we reshape the embeddings accordingly
        transformer_xl_output = self.model(inputs_embeds=encodings)

        # Get the token embeddings from Transformer-XL output
        transformer_xl_embeddings = transformer_xl_output.last_hidden_state

        # Extract the embeddings for each separator token from Transformer-XL output
        separator_embeddings_xl = {}
        for sep in separators:
            positions = separator_positions[sep]
            embeddings = [transformer_xl_embeddings[0, pos, :] for pos in positions] if positions else []
            separator_embeddings_xl[sep] = embeddings

        # Extract the embeddings for each content part from Transformer-XL output
        content_embeddings_xl = []
        for positions in content_positions:
            content_part_embeddings = [transformer_xl_embeddings[0, pos, :] for pos in positions]
            content_embeddings_xl.append(content_part_embeddings)

        del transformer_xl_output
        del encodings
        del separators
        del content_positions
        del contents
        del separator_positions

        gc.collect()
        torch.cuda.empty_cache()
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

        del x
        del attention_scores
        del attention_weights
        gc.collect()
        torch.cuda.empty_cache()
        return attended_output
    

# Function to apply mean pooling and handle padding for a single sentence
def mean_pooling(sequence_output, attention_mask, max_sequence_length):
    # Step 1: Pad the sequence_output to max_sequence_length
    pad_size = max_sequence_length - sequence_output.size(0)
    
    if pad_size > 0:
        # Pad sentence and attention mask
        padded_sentence = torch.cat([sequence_output, torch.zeros((pad_size, sequence_output.size(1)), device=sequence_output.device)], dim=0)
        padded_mask = torch.cat([attention_mask, torch.zeros(pad_size, device=attention_mask.device)], dim=0)
    else:
        padded_sentence = sequence_output
        padded_mask = attention_mask
    
    # Step 2: Create a mean-pooled output that maintains shape (max_seq_len, embedding_dim)
    # input_mask_expanded = padded_mask.unsqueeze(-1).expand(padded_sentence.size()).float()
    
    # Initialize the mean-pooled output
    mean_pooled_output = torch.zeros((max_sequence_length, padded_sentence.size(1)), device=padded_sentence.device)
    
    # Compute mean embeddings for valid tokens
    for i in range(max_sequence_length):
        if i < padded_sentence.size(0) and padded_mask[i] == 1:  # Only consider valid tokens
            mean_pooled_output[i] = padded_sentence[i]  # Assign the embedding directly
        else:
            mean_pooled_output[i] = torch.zeros(padded_sentence.size(1), device=padded_sentence.device)  # Zero padding
    
    del padded_sentence

    gc.collect()
    torch.cuda.empty_cache()
    return mean_pooled_output, padded_mask  # Shape: (max_seq_len, embedding_dim)


def apply_self_attention(context_sentences_emb_list, self_attention_layer):

    # Assuming content_embeddings_xl contains the embeddings for c1, c2, c3, c4
    fixed_length_sentence_vectors = []
    padded_masks = []
    transfoxl_attention_masks = [torch.ones(len(content_embedding)) for content_embedding in context_sentences_emb_list]

    # Step 1: Apply self-attention and mean pooling over each sentence embedding
    max_seq_len = max([len(content_embedding) for content_embedding in context_sentences_emb_list])
    # print(len(context_sentences_emb_list))
    # Step 1: Apply self-attention over each sentence embedding
    for context_sentence, attention_mask in zip(context_sentences_emb_list, transfoxl_attention_masks):
        # print(len(context_sentence))
        context_sentence_embedding_tensor = torch.stack(context_sentence)  # Shape: (seq_len, embedding_dim)
        
        # Apply self-attention over the encoded words of the sentence
        attended_output = self_attention_layer(context_sentence_embedding_tensor.unsqueeze(0))  # Shape: (1, seq_len, embedding_dim)
        
        # Assuming attention_mask for this sentence is available (1 for words, 0 for padding)
        # attention_mask = torch.ones(context_sentence_embedding_tensor.size(0))  # Create mask with 1s for valid tokens

        # Step 2: Apply mean pooling to obtain fixed-length sentence vector
        pooled_output, padded_mask = mean_pooling(attended_output.squeeze(0), attention_mask, max_seq_len) 
        # print(len(pooled_output))
        fixed_length_sentence_vectors.append(pooled_output)
        padded_masks.append(padded_mask)

    del transfoxl_attention_masks
    del context_sentences_emb_list

    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(fixed_length_sentence_vectors), torch.stack(padded_masks)

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
        
        del Q
        del K
        del V
        del attention_scores

        gc.collect()
        torch.cuda.empty_cache()
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

        del pooled_output

        gc.collect()
        torch.cuda.empty_cache()
        return torch.sigmoid(x)  # Use sigmoid for binary classification output

def convert_tensor(tensor_list):
    # Assuming two tensors will form a single matrix row (you can change it based on the requirement)
    flattened_tensors = [tensor.flatten() for tensor in tensor_list]

    # Create a zero tensor of the required shape (2 rows, with a larger max length to pad smaller sequences)
    max_length = max([len(tensor) for tensor in flattened_tensors])
    result = torch.zeros((len(flattened_tensors), max_length))

    # Fill in the values from each tensor into the result matrix
    for i, tensor in enumerate(flattened_tensors):
        result[i, :len(tensor)] = tensor

    del flattened_tensors
    
    gc.collect()
    torch.cuda.empty_cache()
    return result

import torch

def pad_tensor_lists(tensor_list1, tensor_list2):
    # Ensure that both lists have the same number of elements
    assert len(tensor_list1) == len(tensor_list2), "Both lists must have the same number of tensors"
    
    padded_list1 = []
    padded_list2 = []

    # Iterate through each tensor in both lists
    for t1, t2 in zip(tensor_list1, tensor_list2):
        len1 = t1.size(0)
        len2 = t2.size(0)

        # Find the maximum length between the two tensors
        max_len = max(len1, len2)

        # Pad tensors with zeros to match the maximum length
        if len1 < max_len:
            t1 = torch.cat([t1, torch.zeros(max_len - len1, *t1.shape[1:], device=t1.device)], dim=0)
        if len2 < max_len:
            t2 = torch.cat([t2, torch.zeros(max_len - len2, *t2.shape[1:], device=t2.device)], dim=0)
        
        # Add the padded tensors to the output lists
        padded_list1.append(t1.clone().detach().requires_grad_(True))
        padded_list2.append(t2.clone().detach().requires_grad_(True))

    del tensor_list1
    del tensor_list2

    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(padded_list1), torch.stack(padded_list2)

from sklearn.metrics import f1_score

# Step 1: Accuracy Calculation
def calc_accuracy(preds, labels):
    correct = torch.eq(preds, labels).sum().item()  # Count correct predictions
    total = torch.numel(labels)  # Total number of elements in the labels tensor
    gc.collect()
    return correct / total

QID_context = pd.read_pickle('QID_context.pkl')
QID_ans = pd.read_pickle('QID_ans.pkl')
QID_ques = pd.read_pickle('QID_ques.pkl')
QID_q_context = pd.read_pickle('QID_q_context.pkl')
context_qa_list = pd.read_pickle('context_qa_list.pkl')
QID_q_int_type_cont = pd.read_pickle('QID_q_int_type_cont.pkl')


num = 15
labels = []
corr_context = list(QID_context.values())[:num]
corr_ans = list(QID_ans.values())[:num]
for i in range(len(corr_context)):
  context_sent = corr_context[i].split('.')
  ans_sent = corr_ans[i].split('.')
  label = create_labels(context_sent, ans_sent)
  labels.append(label)

# print(labels)

combined_texts = list(QID_q_int_type_cont.values())[:num]

# Split the data
train_texts, temp_texts, train_labels, temp_labels = train_test_split(combined_texts,labels, test_size=0.3, random_state=42)
valid_texts, test_texts, valid_labels, test_labels = train_test_split(temp_texts,temp_labels, test_size=0.5, random_state=42)

print(f"Training set: {len(train_texts)} examples")
print(f"Validation set: {len(valid_texts)} examples")
print(f"Test set: {len(test_texts)} examples")

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_labels)
valid_dataset = CustomDataset(valid_texts, valid_labels)
test_dataset = CustomDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

gc.collect()

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        # Forward pass
        encodings = [framework.forward(text) for text in texts]
        xlnet_encodings, sep, sep_pos, cont, cont_pos = [list(group) for group in list(zip(*encodings))]
        transformed_encodings = [transformer_framework.forward(xlnet_encodings[i], sep[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(xlnet_encodings))]
        transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs = [list(group) for group in list(zip(*transformed_encodings))]
        transfoxl_context_sentences = []
        for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
            context = elem[3:]
            transfoxl_context_sentences.append(context)
        attn_context_sentences = [apply_self_attention(context, self_attention_layer) for context in transfoxl_context_sentences]
        fl_sentence_vectors, padded_masks = [list(group) for group in list(zip(*attn_context_sentences))]
        inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors, padded_masks)]
    
        attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]
    
        predictions = [classification_head(vectors) for vectors in attended_sentence_output]
        logits = convert_tensor([(pred> 0.5).float() for pred in predictions])
        padded_logits, padded_labels = pad_tensor_lists(logits, labels.float())
        padded_labels = padded_labels.to(DEVICE)
        padded_logits = padded_logits.to(DEVICE)
        print(padded_logits)
        print(padded_labels)
        loss = criterion(padded_logits, padded_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        del encodings
        del xlnet_encodings
        del sep
        del sep_pos
        del cont
        del cont_pos
        del transformed_encodings
        del transfoxl_embs
        del transfoxl_sep_embs
        del transfoxl_cont_embs
        del transfoxl_context_sentences
        del attn_context_sentences
        del fl_sentence_vectors
        del padded_masks
        del inter_attended_sentences
        del attended_sentence_output
        del inter_sentence_attention_weights
        del predictions
        del padded_logits
        del padded_labels
        del loss
        del total_loss

    gc.collect()
    torch.cuda.empty_cache()
    return total_loss / len(train_loader)

def evaluate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    accuracy_sample, f1_score_sample = 0, 0
    # with torch.no_grad():
    for texts, labels in tqdm(valid_loader):
        encodings = [framework.forward(text) for text in texts]
        xlnet_encodings, sep, sep_pos, cont, cont_pos = [list(group) for group in list(zip(*encodings))]
        transformed_encodings = [transformer_framework.forward(xlnet_encodings[i], sep[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(xlnet_encodings))]
        transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs = [list(group) for group in list(zip(*transformed_encodings))]
        transfoxl_context_sentences = []
        for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
            context = elem[3:]
            transfoxl_context_sentences.append(context)
        attn_context_sentences = [apply_self_attention(context, self_attention_layer) for context in transfoxl_context_sentences]
        fl_sentence_vectors, padded_masks = [list(group) for group in list(zip(*attn_context_sentences))]
        inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors, padded_masks)]
    
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

        accuracy_sample += calc_accuracy(padded_logits, padded_labels)
        # f1_score_sample += f1_score_per_sample(padded_logits, padded_labels)

    accuracy = (accuracy_sample / len(valid_loader.dataset)) * 100
    gc.collect()
    torch.cuda.empty_cache()
    # f1_score = f1_score_sample / len(valid_loader.dataset)
    return total_loss / len(valid_loader), accuracy

# Define the model, criterion, and optimizer
embedding_dim = 1024 # This is the embeddings dimension of each of transfoxl_embs, can be obtained otherwise by: transfoxl_embs[i].size(-1) 
hidden_size = 512
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
model = torch.nn.DataParallel(model)  # Wrap your model for multi-GPU use
model = model.to(DEVICE)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam([
    {'params': framework.parameters()},
    {'params': transformer_framework.parameters()},
    {'params': self_attention_layer.parameters()},
    {'params': inter_sentence_attention_layer.parameters()},
    {'params': classification_head.parameters()},
], lr=1e-5)


# Training loop
for epoch in tqdm(range(3)):  # Number of epochs
    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_accuracy = evaluate_model(model, valid_loader, criterion)

    print(f'Epoch {epoch+1}/{5}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {valid_loss:.4f}')
    print(f'Validation Accuracy: {valid_accuracy:.4f}')
    # print(f'Validation F1: {valid_f1:.4f}')



