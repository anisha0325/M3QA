
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

# Add [CSEP] tokens in between all context sentences 
def convert_context(context):
    sentences = re.split(r'\.\s*', context)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    csep_sentences = ' [SEP] ' + ' [CSEP] '.join(sentences) + ' [CLS] '
    return csep_sentences

def create_labels(context_sentences, answer_sentences):

    labels = [1 if sentence in answer_sentences else 0 for sentence in context_sentences]
    return labels

class EncodingFramework(nn.Module):
    def __init__(self, model_name='xlnet-base-cased'):
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
        print(separator_ids)

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
        outputs = self.model(input_ids=input_ids)

        # Get the token embeddings (output hidden states)
        token_embeddings = outputs.last_hidden_state

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

        return token_embeddings, separator_positions, content_positions

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

# --------------------------------- STEP 1 -------------------------------------------
''' Create encodings by passing the text through XLNet. Record the positions of the
    seperaters and each element of the input, i.e, question, intent, type and each
    individual sentence of context separately
'''


# Create dictionary QID_q_int_type_cont
QID_q_int_type_cont = dict()

for key, value in tqdm(QID_context.items()):
    converted_context = convert_context(value)
    QID_q_int_type_cont[key] = QID_ques[key] + converted_context

labels = []
corr_context = list(QID_context.values())[:20]
corr_ans = list(QID_ans.values())[:20]
for i in range(len(corr_context)):
  context_sent = corr_context[i].split('.')
  ans_sent = corr_ans[i].split('.')
  label = create_labels(context_sent, ans_sent)
  labels.append(label)

# print(labels)

combined_texts = list(QID_q_int_type_cont.values())[:20]

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

# Define the model, criterion, and optimizer
hidden_size = 768
framework = EncodingFramework()


# a, b, c, = framework.forward(train_texts[0])
for texts, labels in train_loader:
    # Forward pass
    encodings = [framework.forward(text) for text in texts]
    # len(encodings) = batch_size
    # len(encodings[0]) = 3
    # encodings[0][0].shape = torch.Size([1, 1026, 768]) - last_hidden_state
    # encodings[0][1] = position of seperaters
    # encoding[0][2] = position of content entities.
    break




