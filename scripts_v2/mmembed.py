import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import pandas as pd
import re
import requests
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Use your Hugging Face token here

# Load model and tokenizer
model = AutoModel.from_pretrained('nvidia/MM-Embed', trust_remote_code=True).to(device)
print(model.config._name_or_path)

tokenizer = AutoTokenizer.from_pretrained('nvidia/MM-Embed', trust_remote_code=True)

# Define instruction prompt
instruction = "Identify the most relevant sentences from the provided context that answer the given question."

def clean_context(context):
    """Preprocess context: remove special characters, HTML tags, and normalize spaces."""
    context = re.sub(r'<.*?>', '', context)
    context = re.sub(r'[\n\r]+', ' ', context)
    context = re.sub(r'\s+', ' ', context)
    context = re.sub(r'[^\w\s.,]', '', context)
    return context.strip()

def load_images(image_paths):
    """Loads multiple images from given paths or URLs and returns a list of image tensors."""
    images = []
    for image_path in image_paths:
        try:
            if image_path.startswith('http'):  # If it's a URL, download the image
                img = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error loading image: {image_path} - {e}")
    return images

def encode_images(model, images):
    """Encodes multiple images into embeddings and returns their average."""
    if not images:  # No images available
        return None
    with torch.no_grad():
        image_embeddings = [model.encode([{'img': img}])['hidden_states'] for img in images]
        image_embeddings = torch.stack(image_embeddings).mean(dim=0)  # Average embeddings
    return image_embeddings

def find_relevant_sentences(model, context_sentences, question, images, instruction, device='cuda'):
    """
    Finds the most relevant sentences from a context using MM-Embed for text + image retrieval.
    Handles multiple images by averaging their embeddings.
    """
    # queries = [{'txt': question}]
    # if images:  # If images exist, include their embeddings
    #     image_embeddings = encode_images(model, images)
    #     if image_embeddings is not None:
    #         queries[0]['img'] = image_embeddings  # Add images to query

    if images:
        queries = []
        for image in images:
            queries.append({'txt':question, 'img': image})
    else:
        queries = [{'txt': question}]

    passages = [{'txt': sentence} for sentence in context_sentences]
    breakpoint()
    with torch.no_grad():
        query_embeddings = model.encode(queries, is_query=True, instruction=instruction)['hidden_states']
        passage_embeddings = model.encode(passages)['hidden_states']

    # Compute similarity scores
    scores = (query_embeddings @ passage_embeddings.T) * 100  
    scores = scores.squeeze(0).tolist()  # Convert to list

    # Rank sentences by score
    ranked_sentences = sorted(zip(scores, context_sentences), reverse=True, key=lambda x: x[0])

    return [sent for _, sent in ranked_sentences[:5]]  # Top 5 sentences

input_folder = "/nethome/asaha/misc/MedQA_whole/Pytorch_ECAI/"
# Load dataset
QID_context = pd.read_pickle(input_folder + 'QID_context.pkl')
QID_ans = pd.read_pickle(input_folder + 'QID_ans.pkl')
QID_ques = pd.read_pickle(input_folder + 'QID_ques.pkl')

# Image mapping (QID to Image Paths)
image_paths_df = pd.read_csv(input_folder + 'Dataset/question_image_dict.csv')

num = 5 #3012  # Limit number of examples
corr_context = QID_context.iloc[:num].tolist() if isinstance(QID_context, pd.Series) else list(QID_context.values())[:num]
corr_context_keys = list(QID_context.keys())[:num]
corr_ans = QID_ans.iloc[:num].tolist() if isinstance(QID_ans, pd.Series) else list(QID_ans.values())[:num]
corr_ques = QID_ques.iloc[:num].tolist() if isinstance(QID_ques, pd.Series) else list(QID_ques.values())[:num]

results = []
total_questions = len(corr_context)

# Process questions
with tqdm(total=total_questions, desc="Processing Questions", unit="question") as pbar:
    for qid, context, ans, ques in zip(corr_context_keys, corr_context, corr_ans, corr_ques):
        # try:
        # Get corresponding images for QID (multiple images)
        image_paths = image_paths_df[image_paths_df['question_id'] == qid]['image_path'].values
        images = load_images(input_folder + image_paths) if len(image_paths) > 0 else []

        # Clean and split context into sentences
        context_sentences = [clean_context(sentence) for sentence in context.split('.')]
        context_sentences = [sentence.strip() for sentence in context_sentences if sentence]

        # Clean answer sentences
        answer_sentences = [clean_context(sentence) for sentence in ans.split('.')]
        answer_sentences = [sentence.strip() for sentence in answer_sentences if sentence]

        # Find relevant sentences
        predicted_sentences = find_relevant_sentences(model, context_sentences, ques, images, instruction, device=device)

        # Append results
        results.append((qid, predicted_sentences, answer_sentences))

        # except Exception as e:
        #     print(f"Error processing QID {qid}: {e}")

        pbar.update(1)

# Save results
results_df = pd.DataFrame(results, columns=['QID', 'Predicted', 'Actual'])
results_df.to_csv(input_folder + 'MM-Embed_Results_Text_Image.csv', index=False)

def calculate_metrics(results):
    """Calculate accuracy and F1 scores for sentence prediction."""
    all_predicted, all_actual = [], []
    question_level_metrics = []

    for qid, predicted, actual in results:
        predicted_set = set(p.strip() for p in predicted)
        actual_set = set(a.strip() for a in actual)

        question_predicted = [1 if any(a.strip().lower() in p.strip().lower() for p in predicted) else 0 for a in actual]
        question_actual = [1] * len(actual)

        question_accuracy = sum(question_predicted) / len(question_actual) if question_actual else 0
        question_f1 = f1_score(question_actual, question_predicted, average='binary')

        question_level_metrics.append({'qid': qid, 'accuracy': question_accuracy, 'f1_score': question_f1})
        all_predicted.extend(question_predicted)
        all_actual.extend(question_actual)

    return {
        'overall_accuracy': accuracy_score(all_actual, all_predicted),
        'overall_f1_score': f1_score(all_actual, all_predicted, average='binary'),
        'question_level_metrics': question_level_metrics
    }

# Calculate evaluation metrics
metrics = calculate_metrics(results)

print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
print(f"Overall F1 Score: {metrics['overall_f1_score']:.2%}")
