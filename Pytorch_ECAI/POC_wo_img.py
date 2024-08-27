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

random.seed(42)


def create_labels(context_sentences, answer_sentences):

    labels = [1 if sentence in answer_sentences else 0 for sentence in context_sentences]
    return labels

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
        return text

# Define collate function
def collate_fn(batch):
    texts, labels = zip(*batch)
    # texts_padded = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Determine the maximum length of labels in the batch
    max_label_length = max(len(label) for label in labels)

    # Pad labels
    labels_padded = torch.full((len(labels), max_label_length), fill_value=-1, dtype=torch.long)  # Using -1 as padding value
    for i, label in enumerate(labels):
        labels_padded[i, :len(label)] = torch.tensor(label, dtype=torch.long)

    return texts, labels_padded

class EncodingFramework(nn.Module):
    def __init__(self, model_name='xlnet-large-cased'):
        super(EncodingFramework, self).__init__()
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetModel.from_pretrained(model_name)

    def forward(self, text):
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)
        position_ids = list(range(len(tokens)))

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long)
        position_ids = torch.tensor([position_ids], dtype=torch.long)

        outputs = self.model(input_ids=input_ids, token_type_ids=segment_ids, position_ids=position_ids)
        last_hidden_state = outputs.last_hidden_state

        # Return combined encodings
        return last_hidden_state


# class EncodingFramework(nn.Module):
#     def __init__(self, model_name='xlnet-large-cased'):
#         super(EncodingFramework, self).__init__()
#         self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
#         self.model = XLNetModel.from_pretrained(model_name)

#     def forward(self, input_text):
#         # Tokenization
#         tokens = self.tokenizer.tokenize(input_text)
#         input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
#         # Create segment IDs
#         segment_ids = []
#         current_segment = 0
        
#         # Assign segment IDs for Ques, A, B, and Context
#         for token in tokens:
#             if token == '[SEP]':
#                 current_segment += 1
#             segment_ids.append(current_segment)

#         # Prepare position IDs
#         position_ids = list(range(len(tokens)))

#         # Convert to tensors
#         input_ids = torch.tensor([input_ids], dtype=torch.long)
#         segment_ids = torch.tensor([segment_ids], dtype=torch.long)
#         position_ids = torch.tensor([position_ids], dtype=torch.long)

#         # Model forward pass
#         outputs = self.model(input_ids=input_ids, token_type_ids=segment_ids, position_ids=position_ids)
#         last_hidden_state = outputs.last_hidden_state

#         # Create a dictionary to separate out the encodings for each part
#         encoding_dict = {
#             "question": last_hidden_state[0][0],  # Encoding for Ques
#             "a": last_hidden_state[0][1],          # Encoding for A
#             "b": last_hidden_state[0][2],          # Encoding for B
#             "cls": last_hidden_state[0][-1],       # Encoding for [CLS] token
#         }

#         # Extract encodings for context sentences
#         context_encodings = []
#         context_start_idx = 3  # Starting index for context tokens in the encoding
        
#         # Number of context sentences can be determined by the number of [SEP] after Ques, A, and B
#         num_context_sentences = segment_ids.count(3)  # Count the number of segment ID 3 for contexts

#         # Split context by [SEP] and get their encodings
#         for i in range(num_context_sentences):
#             context_encoding = last_hidden_state[0][context_start_idx + i]  # Get the encoding for each context
#             context_encodings.append(context_encoding)

#         # Convert to a tensor of shape [2, 1024]
#         encoding_dict["context"] = torch.stack(context_encodings)  # Stack to create a tensor

#         return encoding_dict


class TransformerXLFramework(nn.Module):
    def __init__(self, model_name='transfo-xl-wt103'):
        super(TransformerXLFramework, self).__init__()
        self.model = TransfoXLModel.from_pretrained(model_name)


    def forward(self, encodings):

        transformer_outputs = self.model(inputs_embeds=encodings)

        return transformer_outputs
    
class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, sentence_embeddings):
        sentence_embeddings = sentence_embeddings.last_hidden_state
        sentence_embeddings = sentence_embeddings.transpose(0, 1)
        attn_output, _ = self.attention(sentence_embeddings, sentence_embeddings, sentence_embeddings)
        attn_output = self.linear(attn_output)  
        return attn_output

class InterSentenceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(InterSentenceAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)

    def forward(self, sentence_vectors):
        sentence_vectors = sentence_vectors.transpose(0, 1)
        attn_output, _ = self.attention(sentence_vectors, sentence_vectors, sentence_vectors)
        return attn_output

class SentenceClassificationHead(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceClassificationHead, self).__init__()
        self.linear = nn.Linear(hidden_size, 2)  # Binary classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sentence_vectors):
        logits = self.linear(sentence_vectors)
        probabilities = self.softmax(logits)
        return probabilities

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        print(texts)
        print()
        print(labels)
        optimizer.zero_grad()
        # Forward pass
        encodings = [framework.forward(text) for text in texts]
        # print(encodings[0]['context'].shape)
        transformed_encodings = [transformer_framework.forward(encoding) for encoding in encodings]
        print([encoding.shape for encoding in transformed_encodings])
        sentence_vectors = [self_attention_layer(encoding) for encoding in transformed_encodings]
        inter_sentence_vectors = [inter_sentence_attention(vectors) for vectors in sentence_vectors]
        logits = [classification_head(vectors) for vectors in inter_sentence_vectors]
        print(logits)
        # print(len(logits))
        # print(type(labels))
        # print(len(labels))
        # Compute loss
        print()
        print(len(labels.tolist()[1]))
        labels_tensor = torch.tensor(labels)
        logits_tensor = torch.tensor(logits)
        print(type(logits_tensor))
        loss = criterion(logits_tensor, labels_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for texts, labels in tqdm(valid_loader):
            encodings = [framework.encode_text(text) for text in texts]
            transformed_encodings = [transformer_framework.forward(encoding) for encoding in encodings]
            sentence_vectors = [self_attention_layer(encoding) for encoding in transformed_encodings]
            inter_sentence_vectors = [inter_sentence_attention(vectors) for vectors in sentence_vectors]
            logits = [classification_head(vectors) for vectors in inter_sentence_vectors]

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(torch.cat(logits), dim=1)
            correct_predictions += (predictions == labels).sum().item()

    accuracy = correct_predictions / len(valid_loader.dataset)
    return total_loss / len(valid_loader), accuracy



QID_context = pd.read_pickle('QID_context.pkl')
QID_ans = pd.read_pickle('QID_ans.pkl')
QID_q_context = pd.read_pickle('QID_q_context.pkl')
QID_ques = pd.read_pickle('QID_ques.pkl')
context_qa_list = pd.read_pickle('context_qa_list.pkl')

labels = []
corr_context = list(QID_context.values())[:500]
corr_ans = list(QID_ans.values())[:500]
for i in range(len(corr_context)):
  context_sent = corr_context[i].split('.')
  ans_sent = corr_ans[i].split('.')
  label = create_labels(context_sent, ans_sent)
  labels.append(label)


combined_texts = list(QID_q_context.values())[:500]

# Split the data
train_texts, temp_texts, train_labels, temp_labels = train_test_split(combined_texts,labels, test_size=0.3, random_state=42)
valid_texts, test_texts, valid_labels, test_labels = train_test_split(temp_texts,temp_labels, test_size=0.5, random_state=42)

print(f"Training set: {len(train_texts)} examples")
print(f"Validation set: {len(valid_texts)} examples")
print(f"Test set: {len(test_texts)} examples")

# Initialize tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')


# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_labels)
valid_dataset = CustomDataset(valid_texts, valid_labels)
test_dataset = CustomDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

# Define the model, criterion, and optimizer
hidden_size = 1024
framework = EncodingFramework()
transformer_framework = TransformerXLFramework()
self_attention_layer = SelfAttentionLayer(hidden_size=hidden_size)
inter_sentence_attention = InterSentenceAttention(hidden_size=hidden_size)
classification_head = SentenceClassificationHead(hidden_size=hidden_size)

model = torch.nn.ModuleList([
    framework, transformer_framework, self_attention_layer, inter_sentence_attention, classification_head
])

criterion = torch.nn.CrossEntropyLoss()  # Example loss function
optimizer = optim.Adam([
    {'params': framework.parameters()},
    {'params': transformer_framework.parameters()},
    {'params': self_attention_layer.parameters()},
    {'params': inter_sentence_attention.parameters()},
    {'params': classification_head.parameters()},
], lr=1e-5)

# Training loop
for epoch in tqdm(range(5)):  # Number of epochs
    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_accuracy = evaluate_model(model, valid_loader, criterion)

    print(f'Epoch {epoch+1}/{5}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {valid_loss:.4f}')
    print(f'Validation Accuracy: {valid_accuracy:.4f}')


