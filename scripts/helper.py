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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import numpy as np
import warnings
import gc
import pickle
import sys #,wandb

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

# Add [CSEP] tokens in between all context sentences 
def convert_context(context):
    sentences = re.split(r'\.\s*', context)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    csep_sentences = ' [SEP] ' + ' [CSEP] '.join(sentences) #change
    gc.collect()
    return csep_sentences

def create_labels(context_sentences, answer_sentences):
    context_sentences = [sentence.strip() for sentence in context_sentences if sentence != '']
    answer_sentences = [sentence.strip() for sentence in answer_sentences if sentence != '']
    labels = [1 if sentence in answer_sentences else 0 for sentence in context_sentences]
    return labels

# Define collate function
def collate_fn_text(batch):
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
def collate_fn_3(batch):
    texts, img, labels = zip(*batch)
    # texts_padded = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Determine the maximum length of labels in the batch
    max_label_length = max(len(label) for label in labels)

    # Pad labels
    labels_padded = torch.full((len(labels), max_label_length), fill_value=0, dtype=torch.long)  # Using -1 as padding value
    for i, label in enumerate(labels):
        labels_padded[i, :len(label)] = torch.tensor(label, dtype=torch.long)
    gc.collect()
    return texts, img, labels_padded

# Define collate function
def collate_fn(batch):
    texts, img, labels, img_labels = zip(*batch)

    # Determine the maximum length of labels in the batch
    max_label_length = max(len(label) for label in labels)
    max_img_label_length = max(len(label) for label in img_labels)
    # Pad labels
    labels_padded = torch.full((len(labels), max_label_length), fill_value=0, dtype=torch.long)  # Using -1 as padding value
    img_labels_padded = torch.full((len(img_labels), max_img_label_length), fill_value=0, dtype=torch.long)
    for i, label in enumerate(labels):
        labels_padded[i, :len(label)] = torch.tensor(label, dtype=torch.long)
    for i, label in enumerate(img_labels):
        img_labels_padded[i, :len(label)] = torch.tensor(label, dtype=torch.long)
    gc.collect()
    return texts, img, labels_padded, img_labels_padded

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


def apply_self_attention(context_sentences_emb_list, self_attention_layer, DEVICE):

    # Assuming content_embeddings_xl contains the embeddings for c1, c2, c3, c4
    fixed_length_sentence_vectors = []
    padded_masks = []
    transfoxl_attention_masks = [torch.ones(len(content_embedding)) for content_embedding in context_sentences_emb_list]
    # print(context_sentences_emb_list)
    # Step 1: Apply self-attention and mean pooling over each sentence embedding
    if len(context_sentences_emb_list) == 0:
        return torch.tensor([]), torch.tensor([])
    else:
        max_seq_len = max([len(content_embedding) for content_embedding in context_sentences_emb_list])
        for context_sentence, attention_mask in zip(context_sentences_emb_list, transfoxl_attention_masks):
            # print(len(context_sentence))
            context_sentence_embedding_tensor = torch.stack(context_sentence).to(DEVICE)  # Shape: (seq_len, embedding_dim)
            
            # Apply self-attention over the encoded words of the sentence
            attended_output = self_attention_layer(context_sentence_embedding_tensor.unsqueeze(0))  # Shape: (1, seq_len, embedding_dim)
            
            # Step 2: Apply mean pooling to obtain fixed-length sentence vector
            pooled_output, padded_mask = mean_pooling(attended_output.squeeze(0), attention_mask, max_seq_len) 
            # print(pooled_output.shape, padded_mask.shape)
            fixed_length_sentence_vectors.append(pooled_output)
            padded_masks.append(padded_mask)
            # except:
            #     pass
        del transfoxl_attention_masks
        del context_sentences_emb_list
        gc.collect()
        torch.cuda.empty_cache()
        return torch.stack(fixed_length_sentence_vectors), torch.stack(padded_masks)



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


def pad_tensor_lists(tensor_list1, tensor_list2):
    # Ensure that both lists have the same number of elements
    assert len(tensor_list1) == len(tensor_list2), "Both lists must have the same number of tensors"
    
    padded_list1 = []
    padded_list2 = []

    # Iterate through each tensor in both lists
    for t1, t2 in zip(tensor_list1, tensor_list2):
        # len1 = t1.size(0)
        # len2 = t2.size(0)

        # # Find the maximum length between the two tensors
        # max_len = max(len1, len2)

        # # Pad tensors with zeros to match the maximum length
        # if len1 < max_len:
        #     t1 = torch.cat([t1, torch.zeros(max_len - len1, *t1.shape[1:], device=t1.device)], dim=0)
        # if len2 < max_len:
        #     t2 = torch.cat([t2, torch.zeros(max_len - len2, *t2.shape[1:], device=t2.device)], dim=0)
        
        min_len = min(t1.size(0), t2.size(0))
        t1 = t1[:min_len]
        t2 = t2[:min_len]
        # Add the padded tensors to the output lists
        padded_list1.append(t1.clone().detach().requires_grad_(True))
        padded_list2.append(t2.clone().detach().requires_grad_(True))

    del tensor_list1
    del tensor_list2

    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(padded_list1), torch.stack(padded_list2)


# Step 1: Accuracy Calculation
def calc_accuracy(labels, preds, type = 'weighted'):
    acc, f1 = 0, 0
    for pred, label in zip(preds, labels):
        acc += accuracy_score(label.cpu(), pred.cpu())
        f1 += f1_score(label.cpu(), pred.cpu(), average=type)
        # print(classification_report(label.cpu(), pred.cpu()))
        # print(classification_report(label.cpu(), pred.cpu()))
    # correct = torch.eq(preds, labels).sum().item()  # Count correct predictions
    # total = torch.numel(labels)
    gc.collect()
    return acc / len(preds), f1 / len(preds)

def log_epoch_info_text_img(file_path, epoch, train_loss, valid_loss, valid_accuracy_sent, valid_accuracy_img, valid_f1_sent, valid_f1_img, test_sent_accuracy, test_sent_f1, test_img_accuracy, test_img_f1):
    # Open the file in append mode ('a') to keep adding lines without overwriting
    file_path = file_path + "epoch_info.txt"
    with open(file_path, 'a') as f:
        f.write(f"Epoch: {epoch}, Train_loss: {train_loss:.4f}, Valid_loss: {valid_loss:.4f}, Valid_sentence_accuracy: {valid_accuracy_sent:.4f}, Valid_image_accuracy: {valid_accuracy_img:.4f}, Test_sentence_accuracy: {test_sent_accuracy:.4f}, Test_image_accuracy: {test_img_accuracy:.4f}, Valid_sentence_f1: {valid_f1_sent:.4f}, Valid_image_f1: {valid_f1_img:.4f}, Test_sentence_f1: {test_sent_f1:.4f}, Test_image_f1: {test_img_f1:.4f}%\n")

def log_epoch_info(file_path, epoch_num, train_loss, valid_loss, valid_sent_accuracy, valid_img_accuracy, test_sent_acc, test_img_acc):
    file_path = file_path + "epoch_info.txt"
    # Open the file in append mode ('a') to keep adding lines without overwriting
    with open(file_path, 'a') as f:
        f.write(f"Epoch: {epoch_num}, Train_loss: {train_loss:.4f}, Valid_loss: {valid_loss:.4f}, Valid_sentence_accuracy: {valid_sent_accuracy:.4f}, Valid_image_accuracy: {valid_img_accuracy:.4f}, Test_sentence_accuracy: {test_sent_acc:.4f}, Test_image_accuracy: {test_img_acc:.4f}%\n")
def log_epoch_info_3(file_path, epoch_num, train_loss, valid_loss, valid_accuracy, valid_f1, test_accuracy, test_f1):

    # Open the file in append mode ('a') to keep adding lines without overwriting
    with open(file_path, 'a') as f:
        f.write(f"Epoch: {epoch_num}, Train_loss: {train_loss:.4f}, Valid_loss: {valid_loss:.4f}, Valid_accuracy: {valid_accuracy:.4f}, Valid_F1: {valid_f1:.4f}, Test_accuracy: {test_accuracy:.4f}, Test_F1: {test_f1:.4f}%\n")

def convert_img_shape(img, image_size, embedding_size, DEVICE):
    # projection_layer = nn.Linear(image_size, embedding_size).to(DEVICE)  # Linear layer to project 768 to 1024
    # projected_image_encoding = projection_layer(img)  # Shape: [1, 1024]
    # Convert to shape [1, 1, 1024]
    projected_image_encoding = img
    if projected_image_encoding.dim() == 1:  # If the tensor is [1024]
        projected_image_encoding = projected_image_encoding.unsqueeze(0).unsqueeze(0)  # Add two dimensions
    elif projected_image_encoding.dim() == 2:  # If the tensor is [1, 1024]
        projected_image_encoding = projected_image_encoding.unsqueeze(0)  # Add one dimension at position 1
    return projected_image_encoding

def combine_text_img_token(text_encoding, projected_image_encoding, cls_encoding, sep_encoding, DEVICE):
    text_encoding = text_encoding.to(DEVICE)
    cls_encoding = cls_encoding.to(DEVICE)
    projected_image_encoding = projected_image_encoding.to(DEVICE)
    sep_encoding = sep_encoding.to(DEVICE)
    combined_tensor = torch.cat((text_encoding, cls_encoding, sep_encoding, projected_image_encoding), dim=1)
    return combined_tensor

def attended_sentence_cls(self_attended_sentence_vectors, transfoxl_sep_embs, mask, DEVICE):
    new_vec, new_mask = [], []
    linear_layer = nn.Linear(2048, 1024).to(DEVICE)
    for i in range(len(self_attended_sentence_vectors)):
        # try:
        tensor_a = self_attended_sentence_vectors[i]
        # breakpoint()
        tensor_b = transfoxl_sep_embs[i]['[CLS]'][0].to(DEVICE)
        n, m, _ = tensor_a.shape
        # Expand tensor_b to match the dimensions of tensor_a for concatenation
        tensor_b_expanded = tensor_b.unsqueeze(0).unsqueeze(0).expand(n, m, -1)

        # Concatenate along the last dimension
        result = torch.cat((tensor_a, tensor_b_expanded), dim=-1)
        new_vec.append(linear_layer(result))


        del tensor_a
        del tensor_b
        del tensor_b_expanded
        del result
    print(len(new_vec))
    return new_vec

def attended_cls_icls(self_attended_sentence_vectors, transfoxl_sep_embs, mask, DEVICE):
    new_vec, new_mask = [], []
    if self_attended_sentence_vectors.numel() == 0:
        return torch.tensor([]).to(DEVICE)
    else:
        # linear_layer = nn.Linear(2048, 1024).to(DEVICE)
        for i in range(len(self_attended_sentence_vectors)):
            # try:
            tensor_a = self_attended_sentence_vectors[i]
            tensor_b = transfoxl_sep_embs['[CLS]'][0].to(DEVICE)
            tensor_c = transfoxl_sep_embs['[ICLS]'][0].to(DEVICE)
            # n, m, _ = tensor_a.shape
            # Expand tensor_b to match the dimensions of tensor_a for concatenation
            tensor_b_expanded = tensor_b.unsqueeze(0)
            tensor_c_expanded = tensor_c.unsqueeze(0)

            # Concatenate along the last dimension
            result = torch.cat((tensor_a, tensor_b_expanded, tensor_c_expanded), dim=0)
            new_vec.append(result)


            del tensor_a
            del tensor_b
            del tensor_b_expanded
            # del result
        # print(len(new_vec))
        # return new_vec
            # print(len(result))
        return torch.stack(new_vec, dim = 0)



import torch
from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel.from_pretrained('xlnet-large-cased')
                                                           
def concat_images(tensor_list, DEVICE):
    # Initialize tokenizer and model

    # Encode special tokens to get their embeddings
    iseparator_token = tokenizer("[ISEP]", add_special_tokens=False, return_tensors="pt").input_ids
    icls_token = tokenizer("[ICLS]", add_special_tokens=False, return_tensors="pt").input_ids

    with torch.no_grad():
        # Get embeddings for [ISEP] and [ICLS]
        isep_embedding = model(iseparator_token).last_hidden_state[:, 0, :].to(DEVICE)  # shape: [1, 1024]
        icls_embedding = model(icls_token).last_hidden_state[:, 0, :].to(DEVICE)        # shape: [1, 1024]

    # Example list of n tensors, each of shape [1, 1024]
    n = len(tensor_list)

    # Build the sequence with tensors and [ISEP], ending with [ICLS]
    sequence = []

    for tensor in tensor_list:
        sequence.append(tensor.to(DEVICE))
        sequence.append(isep_embedding)

    # Remove the last [ISEP] and add [ICLS] at the end
    sequence = sequence[:-1] + [icls_embedding]
    # Concatenate into one tensor of shape [(n * 2) - 1, 1024]
    final_sequence = torch.cat(sequence, dim=0)
    return final_sequence

def only_one_loss(labels,logits, DEVICE):
    one_labels, one_logits = [], []
    for label, logit in zip(labels, logits):
        positions = torch.nonzero(label == 1.0, as_tuple=True)
        one_label = label[positions]
        one_logit = logit[positions]
        one_labels.append(one_label.clone().detach().requires_grad_(True))
        one_logits.append(one_logit.clone().detach().requires_grad_(True))
    # Pad sequences to the same length
    padded_predictions = pad_sequence(one_logits, batch_first=True)  # Shape: (4, max_len)
    padded_targets = pad_sequence(one_labels, batch_first=True)      # Shape: (4, max_len)
    # Create a mask to identify valid positions
    # Create a mask to identify valid positions
    lengths = torch.tensor([len(t) for t in padded_targets], device=DEVICE)  # Original lengths
    mask = torch.arange(padded_targets.size(1), device=DEVICE).expand(len(lengths), -1) < lengths.unsqueeze(1)

    # Compute the loss while ignoring padded positions
    loss = F.binary_cross_entropy(
        padded_predictions, 
        padded_targets.float(), 
        reduction="none"
    )

    # Apply mask and compute mean loss over valid positions
    loss = (loss * mask).sum() / mask.sum()
    return one_labels, one_logits, loss

def gen_answer(logits, labels, texts):
    pred_answers = []
    true_answers = []
    questions = []
    for logit, label, text in zip(logits, labels, texts): 
        # Split the paragraph by the [CSEP] token to get dynamic contents
        all_separated = text.split("[SEP]")
        question = all_separated[0]
        all_separated = [content.strip() for content in all_separated if content.strip()]  # Remove any extra spaces

        no_context = all_separated[:-1]

        only_context = all_separated[-1]
        only_context_items = only_context.split("[CSEP]")
        only_context_items = [content.strip() for content in only_context_items if content.strip()] 

        # Predicted answer
        sentence_list = only_context_items[:len(logit)+1]
        selected_indices = torch.nonzero(logit == 1.0, as_tuple=True)[0].tolist()
        if len(selected_indices) == 0:
            new_sentences = []
        else:
            new_sentences = [sentence_list[i] for i in selected_indices]   
        pred_answers.append(''.join(new_sentences))

        # True answer
        sentence_list = only_context_items[:len(label)+1]
        selected_indices = torch.nonzero(label == 1.0, as_tuple=True)[0].tolist()
        if len(selected_indices) == 0:
            new_sentences = []
        else:
            new_sentences = [sentence_list[i] for i in selected_indices]  
        true_answers.append(''.join(new_sentences))

        questions.append(question)

    return pred_answers, true_answers, questions