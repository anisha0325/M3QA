from transformers import AutoTokenizer, BitsAndBytesConfig
from LLaVA.llava.model import LlavaLlamaForCausalLM
import torch
from PIL import Image
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.mm_utils import tokenizer_image_token
from LLaVA.llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import os
import pickle
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def safe_image_retrieval(image_paths, qid):
    """
    Safely retrieve image files for a given question ID.
    
    Args:
    image_paths (pd.DataFrame): DataFrame containing image paths
    qid (str): Question ID to search for
    
    Returns:
    list: Image file paths or empty list if not found
    """
    try:
        # Use .loc with a fallback to check for matching question_id
        matching_rows = image_paths[image_paths['question_id'] == qid]
        
        if len(matching_rows) > 0:
            # Try to evaluate the image path
            try:
                image_files = eval(matching_rows['image_path'].values[0])
                return image_files
            except (SyntaxError, ValueError) as e:
                print(f"Error evaluating image path for QID {qid}: {e}")
                return []
        else:
            print(f"No image files found for QID {qid}")
            return []
    
    except Exception as e:
        print(f"Unexpected error retrieving images for QID {qid}: {e}")
        return []


# Model Path
model_path = "4bit/llava-v1.5-13b-3GB"

# Load Model with Quantization
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load Vision Tower
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

# Preprocess Images
def preprocess_images(image_files):
    # Define supported image extensions
    supported_extensions = {'.jpg', '.png', '.jpeg', '.webp', '.avif', '.gif', '.svg', '.JPG'}
    
    image_tensors = []
    for image_file in image_files:
        # Check if the file extension is supported
        if os.path.splitext(image_file)[-1] in supported_extensions:
            try:
                image = Image.open(image_file).convert('RGB')
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
                image_tensors.append(image_tensor)
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
        else:
            print(f"Unsupported file format: {image_file}")
    
    if image_tensors:
        # Concatenate all image tensors and calculate the mean tensor
        concatenated_tensors = torch.cat(image_tensors, dim=0)
        mean_tensor = concatenated_tensors.mean(dim=0, keepdim=True)
        return mean_tensor
    else:
        raise ValueError("No valid images found to process.")

# Generate Relevant Sentences
def find_relevant_sentences(context_sentences, question, image_files):
    # Preprocess images
    image_tensors = preprocess_images(image_files)

    # Prepare conversation template
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    # Combine context and question
    context = "\n".join([f"{idx + 1}. {sentence}" for idx, sentence in enumerate(context_sentences)])
    inp = (f"You are given a set of context sentences and images. Your task is to identify which sentences "
           f"in the context are most relevant to the question based on both textual and visual information.\n"
           f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nTASK: "
           f"List the sentences that are most relevant to answering the question.\n")
    # Add image tokens
    inp = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n" + inp

    # Append messages
    roles = conv.roles
    conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)
    raw_prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Generate output
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            images=image_tensors,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # Check and filter token IDs
    token_ids = outputs[0].tolist()
    valid_ids = [token_id for token_id in token_ids if 0 <= token_id < tokenizer.vocab_size]
    
    # Decode and process output
    decoded_output = tokenizer.decode(valid_ids, skip_special_tokens=True)
    return decoded_output



def clean_context(context):
    # Remove non-natural language content such as HTML tags and code
    context = re.sub(r'<.*?>', '', context)  # Remove HTML tags
    context = re.sub(r'[\n\r]+', ' ', context)  # Remove newlines
    context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
    context = re.sub(r'[^\w\s.,]', '', context)  # Remove special characters (keep punctuation)
    return context.strip()

QID_context = pd.read_pickle('./Pytorch_ECAI/QID_context.pkl')
QID_ans = pd.read_pickle('./Pytorch_ECAI/QID_ans.pkl')
QID_ques = pd.read_pickle('./Pytorch_ECAI/QID_ques.pkl')
QID_q_context = pd.read_pickle('./Pytorch_ECAI/QID_q_context.pkl')
context_qa_list = pd.read_pickle('./Pytorch_ECAI/context_qa_list.pkl')
QID_q_int_type_cont = pd.read_pickle('./Pytorch_ECAI/QID_q_int_type_cont.pkl')
image_paths = pd.read_csv('./Pytorch_ECAI/Dataset/question_image_dict_list.csv')

num=3012
corr_context = list(QID_context.values())[:num]
corr_context_keys = list(QID_context.keys())[:num]
corr_ans = list(QID_ans.values())[:num]
corr_ques = list(QID_ques.values())[:num]

results = []
total_questions = len(corr_context)  

with tqdm(total=total_questions, desc="Processing Questions", unit="question") as pbar:
    for qid, context, ans, ques in zip(corr_context_keys, corr_context, corr_ans, corr_ques):
        try:
            # Clean context and answer sentences
            context_sentences = [clean_context(sentence) for sentence in context.split('.')]
            context_sentences = [sentence.strip() for sentence in context_sentences if sentence]
            
            answer_sentences = [clean_context(sentence) for sentence in ans.split('.')]
            answer_sentences = [sentence.strip() for sentence in answer_sentences if sentence]
            
            # Safe image retrieval
            image_files = safe_image_retrieval(image_paths, qid)
            
            # Skip if no image files found
            if not image_files:
                print(f"Skipping QID {qid} due to no image files")
                pbar.update(1)
                continue
            
            # Process relevant sentences
            predicted_sentences = find_relevant_sentences(context_sentences, ques, image_files).split("Assistant:")[1]
            predicted_sentences = " ".join(predicted_sentences.split())
            predicted_sentences = predicted_sentences.split(". ")
            predicted_sentences = [sentence.strip() for sentence in predicted_sentences if sentence]
            
            # Append results
            results.append((qid, predicted_sentences, answer_sentences))
        
        except Exception as e:
            print(f"Error processing QID {qid}: {e}")
        
        # Always update progress bar
        pbar.update(1)

# Optional: Log skipped or problematic questions
skipped_questions = total_questions - len(results)
print(f"Processed {len(results)} out of {total_questions} questions")
print(f"Skipped {skipped_questions} questions")

# Save results
results_df = pd.DataFrame(results, columns=['QID', 'Predicted', 'Actual'])
results_df.to_csv('LlaVA_Results.csv', index=False)

def calculate_metrics(results):
    """
    Calculate accuracy and F1 scores for sentence prediction.
    
    Args:
    results (list): List of tuples containing (qid, predicted_sentences, actual_sentences)
    
    Returns:
    dict: Dictionary containing overall metrics
    """
    # Prepare lists for metrics calculation
    all_predicted = []
    all_actual = []
    
    # Per-question metrics tracking
    question_level_metrics = []
    
    for qid, predicted, actual in results:
        # Normalize sentences by stripping and converting to sets
        predicted_set = set(p.strip() for p in predicted)
        actual_set = set(a.strip() for a in actual)
        
        # Metrics for this specific question
        # Binary classification: was each actual sentence predicted?
        question_predicted = []
        question_actual = []
        
        for sentence in actual:
            # Check if any predicted sentence matches this actual sentence
            match_found = any(
                sentence.strip().lower() in pred.strip().lower() or 
                pred.strip().lower() in sentence.strip().lower() 
                for pred in predicted
            )
            
            question_predicted.append(1 if match_found else 0)
            question_actual.append(1)
        
        # Calculate metrics for this question
        question_accuracy = sum(question_predicted) / len(question_actual) if question_actual else 0
        question_f1 = f1_score(question_actual, question_predicted, average='binary')
        
        question_level_metrics.append({
            'qid': qid,
            'accuracy': question_accuracy,
            'f1_score': question_f1
        })
        
        # Aggregate for overall metrics
        all_predicted.extend(question_predicted)
        all_actual.extend(question_actual)
    
    # Overall metrics
    overall_accuracy = accuracy_score(all_actual, all_predicted)
    overall_f1 = f1_score(all_actual, all_predicted, average='binary')
    
    return {
        'overall_accuracy': overall_accuracy,
        'overall_f1_score': overall_f1,
        'question_level_metrics': question_level_metrics
    }

# Modify the existing code to use this evaluation function
# Replace the existing accuracy calculation with:
metrics = calculate_metrics(results)

print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
print(f"Overall F1 Score: {metrics['overall_f1_score']:.2%}")

# Optional: Save detailed metrics
metrics_df = pd.DataFrame(metrics['question_level_metrics'])
metrics_df.to_csv('question_level_metrics.csv', index=False)

# Save results with more detailed information
results_df = pd.DataFrame(results, columns=['QID', 'Predicted', 'Actual'])
results_df['Metrics'] = metrics_df.to_dict('records')
results_df.to_csv('comprehensive_results.csv', index=False)
