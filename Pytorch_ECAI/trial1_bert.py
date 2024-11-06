
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from transformers import XLNetTokenizer, XLNetModel, TransfoXLModel
from transformers import BertTokenizer, BertModel
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
import sys #,wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

# 5 --> 500 (new)
# 4 --> 500 (old)
# 6 --> 1000 
# 2 --> img

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
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
    DEVICE = torch.device("cuda")
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

# class EncodingFramework(nn.Module):
#     def __init__(self, model_name='xlnet-large-cased'):
#         super(EncodingFramework, self).__init__()
#         self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
#         self.model = XLNetModel.from_pretrained(model_name)
class EncodingFramework(nn.Module):
    def __init__(self, model_name='bert-large-cased'):
        super(EncodingFramework, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, text):
        # Tokenize the input while keeping special tokens intact
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[CSEP]', '[SEP]', '[CLS]']})
        self.model.resize_token_embeddings(len(self.tokenizer))

        tokens = self.tokenizer(text, add_special_tokens=True, return_tensors="pt",max_length=512,truncation=True,padding='max_length')
        # print(tokens['input_ids'].shape)
        # Get tokenized IDs
        input_ids = tokens['input_ids']#[:,:512]
        # print(f"Token IDs: {input_ids}")
        # Pass the tokenized input through the XLNet model to get the embeddings
        input_ids = input_ids.to(DEVICE)
        # print("Before Processing:")
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
        # with torch.no_grad():
        outputs = self.model(input_ids=input_ids)
        # print('0.3')
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
        # Get the token embeddings (output hidden states)

        token_embeddings = outputs.last_hidden_state.detach().cpu()
        # print(token_embeddings.device)
        # print('1')
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
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

        # # Extract the embeddings for each separator token
        # separator_embeddings = {}
        # for sep in separators:
        #     positions = separator_positions[sep]
        #     embeddings = [token_embeddings[0, pos, :] for pos in positions] if positions else []
        #     separator_embeddings[sep] = embeddings

        # # Extract the embeddings for each content part (separately)
        # content_embeddings = []
        # for positions in content_positions:
        #     content_part_embeddings = [token_embeddings[:, pos, :] for pos in positions]
        #     content_embeddings.append(content_part_embeddings)

        # # Print out positions and shapes of embeddings for separator tokens
        # for sep in separators:
        #     positions = separator_positions[sep]
        #     embeddings = separator_embeddings[sep]
        #     for i, embedding in enumerate(embeddings):
        #         print(f"Position of '{sep}': {positions[i]}, Embedding shape: {embedding.shape}")

        # # Print out positions and shapes of embeddings for each content part
        # for i, (positions, embeddings) in enumerate(zip(content_positions, content_embeddings)):
        #     print(f"Content part {i+1} positions: {positions}")
        #     for j, embedding in enumerate(embeddings):
        #         print(f"  Token {j+1} at position {positions[j]}: Embedding shape: {embedding.shape}")
        del input_ids
        del outputs
        del tokens
        del all_separated
        del no_context
        del only_context
        del only_context_items
        del content_ids
        del separator_ids
        del positions

        token_embeddings_cpu = token_embeddings.cpu()
        del token_embeddings
        # check_mem(token_embeddings)
        gc.collect()
        torch.cuda.empty_cache()
        # print('5')
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
        return token_embeddings_cpu, separators, separator_positions, contents, content_positions

def check_mem(gpu_tensor):
    if torch.cuda.is_available():
        memory_in_bytes = gpu_tensor.numel() * gpu_tensor.element_size()

        # Convert bytes to MB
        memory_in_MB = memory_in_bytes / (1024 ** 2)

        print(f"Memory occupied by the tensor on GPU: {memory_in_MB:.2f} MB")
    else:
        print("CUDA not available. Tensor is on CPU.")

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
        # print("Inside TransformerXLFramework")
        # # Transformer-XL expects input in 2D format, so we reshape the embeddings accordingly
        # transformer_xl_output = self.model(inputs_embeds=encodings)
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")

        # # Get the token embeddings from Transformer-XL output
        # transformer_xl_embeddings = transformer_xl_output.last_hidden_state
        # print(encodings.shape)
        total_len = encodings.size(1)
        transformer_xl_embeddings = []
        memory = None  # Initialize memory to None for the first segment
        chunk_size = 512
        # print(total_len)
        # Process each encoding in chunks
        for i in range(0, total_len, chunk_size):
            # print()
            # print(i)
            # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
            # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
            # Extract chunk of input
            chunk_encodings = encodings[:, i:i + chunk_size].to(DEVICE)  # Move chunk to GPU

            # Pass chunk through Transformer-XL, along with memory from the previous chunk
            transformer_xl_output = self.model(inputs_embeds=chunk_encodings, mems=memory)

            # Get the token embeddings from Transformer-XL output
            transformer_xl_embeddings.append(transformer_xl_output.last_hidden_state.detach().cpu())  # Move output to CPU to save GPU memory

            # Update memory with the current chunk's memory, move memory to GPU for next iteration
            memory = [mem.to(DEVICE) for mem in transformer_xl_output.mems]

            # Clear the chunk_encodings and output from GPU memory after use
            del chunk_encodings, transformer_xl_output
            torch.cuda.empty_cache()  # Free unused memory

        # Concatenate all chunk embeddings along the sequence dimension
        transformer_xl_embeddings = torch.cat(transformer_xl_embeddings, dim=1)
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

        # # Print out positions and shapes of embeddings for separator tokens from Transformer-XL
        # for sep in separators:
        #     positions = separator_positions[sep]
        #     embeddings = separator_embeddings_xl[sep]
        #     for i, embedding in enumerate(embeddings):
        #         print(f"Position of '{sep}' in Transformer-XL: {positions[i]}, Embedding shape: {embedding.shape}")

        # # Print out positions and shapes of embeddings for each content part from Transformer-XL
        # for i, (positions, embeddings) in enumerate(zip(content_positions, content_embeddings_xl)):
        #     print(f"Content part {i+1} positions in Transformer-XL: {positions}")
        #     for j, embedding in enumerate(embeddings):
        #         print(f"  Token {j+1} at position {positions[j]}: Embedding shape: {embedding.shape}")

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
    
# Function to apply mean pooling
# def mean_pooling(sequence_output, attention_mask):
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
#     sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return sum_embeddings / sum_mask

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
    input_mask_expanded = padded_mask.unsqueeze(-1).expand(padded_sentence.size()).float()
    
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
        try:
            # print(len(context_sentence))
            context_sentence_embedding_tensor = torch.stack(context_sentence).to(DEVICE)  # Shape: (seq_len, embedding_dim)
            
            # Apply self-attention over the encoded words of the sentence
            attended_output = self_attention_layer(context_sentence_embedding_tensor.unsqueeze(0))  # Shape: (1, seq_len, embedding_dim)
            
            # Assuming attention_mask for this sentence is available (1 for words, 0 for padding)
            # attention_mask = torch.ones(context_sentence_embedding_tensor.size(0))  # Create mask with 1s for valid tokens

            # Step 2: Apply mean pooling to obtain fixed-length sentence vector
            pooled_output, padded_mask = mean_pooling(attended_output.squeeze(0), attention_mask, max_seq_len) 
            # print(len(pooled_output))
            fixed_length_sentence_vectors.append(pooled_output)
            padded_masks.append(padded_mask)
        except:
            pass
    # Step 3: Convert the list of sentence vectors into a tensor
    # fixed_length_sentence_vectors = torch.stack(fixed_length_sentence_vectors)  # Shape: (num_sentences, embedding_dim)

    # # Output: fixed_length_sentence_vectors contains a fixed-length vector for each sentence
    # for i, sentence_vector in enumerate(fixed_length_sentence_vectors):
    #     print(f"Fixed-length vector for sentence {i+1}: Shape {sentence_vector.shape}")

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
            # print("Mask shape: ", mask.shape)
            # print("Attention_scores shape: ",attention_scores.shape)
            # print("Mask expanded shape: ", mask_expanded.shape)
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

def log_epoch_info(file_path, epoch_num, train_loss, valid_loss, valid_accuracy):

    # Open the file in append mode ('a') to keep adding lines without overwriting
    with open(file_path, 'a') as f:
        f.write(f"Epoch: {epoch_num}, Train_loss: {train_loss:.4f}, Valid_loss: {valid_loss:.4f}, Valid_accuracy: {valid_accuracy:.2f}%\n")

def attended_sentence_cls(self_attended_sentence_vectors, transfoxl_sep_embs, mask):
    new_vec, new_mask = [], []
    linear_layer = nn.Linear(2048, 1024).to(DEVICE)
    for i in range(len(self_attended_sentence_vectors)):
        try:
            tensor_a = self_attended_sentence_vectors[i]
            tensor_b = transfoxl_sep_embs[i]['[CLS]'][0].to(DEVICE)
            n, m, _ = tensor_a.shape
            # Expand tensor_b to match the dimensions of tensor_a for concatenation
            tensor_b_expanded = tensor_b.unsqueeze(0).unsqueeze(0).expand(n, m, -1)

            # Concatenate along the last dimension
            result = torch.cat((tensor_a, tensor_b_expanded), dim=-1)
            new_vec.append(linear_layer(result))

            # Extend the mask with an additional 1 for the CLS token
            # mask_i = mask[i]
            # cls_mask = torch.ones((1, mask_i.shape[1]), dtype=mask_i.dtype, device=mask_i.device)
            # extended_mask = torch.cat((mask_i, cls_mask), dim=0) 
            # new_mask.append(extended_mask)

            del tensor_a
            del tensor_b
            del tensor_b_expanded
            del result
            # del mask_i
            # del cls_mask
            # del extended_mask
        except:
            print("In Except")
            continue
    print(len(new_vec))
    # return new_vec, new_mask
    return new_vec
        

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    c, num_excl = 0, 0
    for texts, labels in tqdm(train_loader):
        optimizer.zero_grad()
        # print("START")
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
        # Forward pass
        encodings = [framework.forward(text) for text in texts]
        # print("After EncodingFramework")
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
        xlnet_encodings, sep, sep_pos, cont, cont_pos = zip(*encodings) #[list(group) for group in list(zip(*encodings))]
        # for enc in xlnet_encodings:
        #     check_mem(enc)
        # xlnet_encodings_cpu = [elem.to('cpu') for elem in xlnet_encodings]
        del encodings
        gc.collect()
        # torch.cuda.empty_cache()
        # print(xlnet_encodings_cpu[0].device)
        # print(type(sep[0]))
        transformed_encodings = [transformer_framework.forward(xlnet_encodings[i].to(DEVICE), sep[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(xlnet_encodings))]
        transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs = [list(group) for group in list(zip(*transformed_encodings))]
        transfoxl_context_sentences = []
        for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
            context = elem[3:]
            transfoxl_context_sentences.append(context)
        # print("After TrandformerXL")
        # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
        attn_context_sentences = [apply_self_attention(context, self_attention_layer) for context in transfoxl_context_sentences]
        fl_sentence_vectors, padded_masks = [list(group) for group in list(zip(*attn_context_sentences))]
        # fl_sentence_vectors_cls, padded_masks_cls = attended_sentence_cls(fl_sentence_vectors, transfoxl_sep_embs, padded_masks)
        fl_sentence_vectors_cls= attended_sentence_cls(fl_sentence_vectors, transfoxl_sep_embs, padded_masks)
        if len(fl_sentence_vectors_cls) == batch_size:
            # inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
            #                         for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks_cls)]
            inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                    for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks)]
            attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]
            # print("After Attentions")
            # print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2)} MB")
            # print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2)} MB")
            # print("END of Iter")
            predictions = [classification_head(vectors) for vectors in attended_sentence_output]
            logits = convert_tensor([(pred> 0.5).float() for pred in predictions])
            padded_logits, padded_labels = pad_tensor_lists(logits, labels.float())
            padded_labels = padded_labels.to(DEVICE)
            padded_logits = padded_logits.to(DEVICE)
            # print(padded_logits)
            # print(padded_labels)
            loss = criterion(padded_logits, padded_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            del inter_attended_sentences
            del attended_sentence_output
            del inter_sentence_attention_weights
            del logits
            del labels
            del predictions
            del padded_logits
            del padded_labels
            del loss
            # wandb.log({"batch_loss": loss.item()})
        else:
            num_excl = num_excl + 1
            print("Batch {} excluded".format(c))
            with open("bath_excluded.txt", 'a') as f:
                f.write(f"Batch: {c} excluded from training %\n")
        del texts
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
        gc.collect()
        torch.cuda.empty_cache()
        # wandb.log({"epoch_train_loss": train_loss})
        c = c + 1
    return total_loss / (len(train_loader) - num_excl)

def evaluate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    accuracy_sample, f1_score_sample = 0, 0
    c, num_excl = 0,0 
    with torch.no_grad():
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
            fl_sentence_vectors_cls= attended_sentence_cls(fl_sentence_vectors, transfoxl_sep_embs, padded_masks)
            if len(fl_sentence_vectors_cls) == batch_size:
                inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
                                    for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors_cls, padded_masks)]
        
                attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]
        
                predictions = [classification_head(vectors) for vectors in attended_sentence_output]
                logits = convert_tensor([(pred> 0.5).float() for pred in predictions])
                padded_logits, padded_labels = pad_tensor_lists(logits, labels.float())
                padded_labels = padded_labels.to(DEVICE)
                padded_logits = padded_logits.to(DEVICE)
                loss = criterion(padded_logits, padded_labels)
            # loss.backward()
            # optimizer.step()

                total_loss += loss.item()

            # Compute accuracy
            # predictions = torch.argmax(torch.cat(logits), dim=1)
            # correct_predictions += (predictions == labels).sum().item()
                accuracy_sample += calc_accuracy(padded_logits, padded_labels)
            # f1_score_sample += f1_score_per_sample(padded_logits, padded_labels)

                del inter_attended_sentences
                del attended_sentence_output
                del inter_sentence_attention_weights
                del predictions
                del padded_logits
                del padded_labels
                del loss
            else:
                num_excl = num_excl + 1
                print("Batch {} excluded from eval".format(c))
                with open("batch_excluded.txt", 'a') as f:
                    f.write(f"Batch: {c} excluded from eval %\n")


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

            gc.collect()
            torch.cuda.empty_cache()


    gc.collect()
    torch.cuda.empty_cache()
    c = c + 1
    # f1_score = f1_score_sample / len(valid_loader.dataset)
    # wandb.log({"epoch_valid_loss": valid_loss, "epoch_valid_accuracy": valid_accuracy})
    return total_loss / (len(valid_loader) - num_excl), accuracy_sample / (len(valid_loader) - num_excl)

# # Step 2: F1 Score Calculation (multi-label, per sample)
# def f1_score_per_sample(preds, labels):
#     preds_np = preds.detach().numpy()  # Convert tensors to NumPy arrays
#     labels_np = labels.detach().numpy()
    
#     # Calculate F1 score for each list of predictions and true labels
#     f1_scores = []
#     for pred, true in zip(preds_np, labels_np):
#         f1 = f1_score(true, pred, average='binary')  # Binary F1 score for each list
#         f1_scores.append(f1)
        
#     return f1_scores


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

# --------------------------------- STEP 1 -------------------------------------------
''' Create encodings by passing the text through XLNet. Record the positions of the
    seperaters and each element of the input, i.e, question, intent, type and each
    individual sentence of context separately
'''


# Create dictionary QID_q_int_type_cont
# QID_q_int_type_cont = dict()

# for key, value in tqdm(QID_context.items()):
#     converted_context = convert_context(value)
#     QID_q_int_type_cont[key] = QID_ques[key] + converted_context

# with open('QID_q_int_type_cont.pkl', 'wb') as file:
#     pickle.dump(QID_q_int_type_cont, file)



# num = len(list(QID_context.values()))
num = 1000
print("Total No. Of Datapoints: ", num)
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

# print(labels)

combined_texts = list(QID_q_int_type_cont.values())[:num]
# combined_texts = list(QID_q_int_type_cont.values())

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

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

gc.collect()

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

# a, b, c, = framework.forward(train_texts[0])
# for texts, labels in train_loader:
#     # Forward pass

#     ''' 
#     len(encodings) = batch_size
#     len(encodings[0]) = 5 (retuns 5 elements)
#     encodings[0][0].shape = torch.Size([1, 936, 1024]) - last_hidden_state
#     encodings[0][1] = position of seperaters
#     encodings[0][2] = encodings of seperators
#     encoding[0][3] = position of content entities.
#     encodings[0][4] = encodings of content entities
#     Segregating this list in the form [enc1, enc2,...encn], [sep1,...,sepn], 
#     [pos_sep1,...,pos_sepn], [cont1,...,contn], [pos_cont1,...,post_contn]

#     '''
#     encodings = [framework.forward(text) for text in texts]
#     xlnet_encodings, sep, sep_pos, cont, cont_pos = [list(group) for group in list(zip(*encodings))]

#     ''' 
#     len(transformed_encodings) = batch_size
#     len(transformed_encodings[0]) = 3
#     transformed_encodings[0][0].shape = torch.Size([1, 936, 1024]) - last_hidden_state
#     transformed_encodings[0][1] = embeddings of seperaters from transformer-xl
#     transformed_encodings[0][2] = embeddings of content entities from transformer-xl
#     Segregating this list in the form [enc1, enc2,...encn], [sep1,...,sepn], 
#     [pos_sep1,...,pos_sepn], [cont1,...,contn], [pos_cont1,...,post_contn]

#     len(transfoxl_cont_embs) = 2 (batch_size)
#     transfoxl_cont_embs[i] = Represents each of question, intent, type, context sentences respectively
#     transfoxl_cont_embs[i][0] = question
#     transfoxl_cont_embs[i][1] = intent
#     transfoxl_cont_embs[i][2] = type
#     transfoxl_cont_embs[i][3] to transfoxl_cont_embs[i][n] = Each represent a context sentence
    
#     '''

#     transformed_encodings = [transformer_framework.forward(xlnet_encodings[i], sep[i], sep_pos[i], cont[i], cont_pos[i]) for i in range(len(xlnet_encodings))]
#     transfoxl_embs, transfoxl_sep_embs, transfoxl_cont_embs = [list(group) for group in list(zip(*transformed_encodings))]
#     transfoxl_context_sentences = []
#     for elem in transfoxl_cont_embs:# Third position onwards are the context sentences
#         context = elem[3:]
#         transfoxl_context_sentences.append(context)
#     attn_context_sentences = [apply_self_attention(context, self_attention_layer) for context in transfoxl_context_sentences]
#     fl_sentence_vectors, padded_masks = [list(group) for group in list(zip(*attn_context_sentences))]
#     inter_attended_sentences = [inter_sentence_attention_layer(final_sentence_embedding, padded_mask)
#                                 for (final_sentence_embedding, padded_mask) in zip(fl_sentence_vectors, padded_masks)]
    
#     attended_sentence_output, inter_sentence_attention_weights = [list(group) for group in list(zip(*inter_attended_sentences))]
#     predictions = [classification_head(elem) for elem in attended_sentence_output]
#     binary_predictions = [(pred> 0.5).float() for pred in predictions]
#     # breakpoint()
    

# wandb.login()
# wandb.init(project='MEDQA',)
# wandb.config = {
#   "learning_rate": 1e-5,
#   "epochs": 10,
#   "batch_size": 4
# }

# Training loop
file_path = "trial1-1000-25.txt"
EPOCHS = 25
for epoch in tqdm(range(EPOCHS)):  # Number of epochs
    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_accuracy = evaluate_model(model, valid_loader, criterion)

    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {valid_loss:.4f}')
    print(f'Validation Accuracy: {valid_accuracy:.4f}')

    log_epoch_info(file_path, epoch, train_loss, valid_loss, valid_accuracy)
    # print(f'Validation F1: {valid_f1:.4f}')

# wandb.finish()

