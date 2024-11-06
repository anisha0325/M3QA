import csv
import ast
import pickle
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
from torch.nn.functional import cosine_similarity
import torch
from tqdm import tqdm
import glob

# Load the questions data
with open('QID_ques.pkl', 'rb') as f:
    questions_data = pickle.load(f)  # This should contain the question text for each question_id

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Check if CUDA is available and move model to GPU
if torch.cuda.is_available():
    model = model.cuda()
    print("Using CUDA")

# Read the CSV file with question_id and relevant image paths
with open('question_image_dict.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    # Prepare CSV file for output
    with open('cosine_similarity_results.csv', 'w', newline='') as outputfile:
        csv_writer = csv.writer(outputfile)
        csv_writer.writerow(["Question ID", "Question Text", "Image Path", "Image Number", "Cosine Similarity", "Mapped"])

        # Process each row in the input CSV
        for row in tqdm(reader, desc="Processing Questions"):
            q_id = row['question_id']
            image_paths = ast.literal_eval(row['image_path'])  # Convert string representation of list to actual list

            # Check if the question text exists for this question_id
            if q_id not in questions_data:
                tqdm.write(f"Skipping question ID {q_id} due to missing question text.")
                continue

            question_text = questions_data[q_id]
            text = question_text.split('[SEP]')[0]  # Split question text into parts if needed

            # Get the folder path from the first image path
            if image_paths:
                folder_path = os.path.dirname(image_paths[0])  # e.g., "Dataset/Annotations/F01"

                # Collect all images in this folder
                all_images = glob.glob(os.path.join(folder_path, "*.*"))  # Get all images in the folder

                # Filter out non-mapped images by comparing with image_paths
                additional_images = set(all_images) - set(image_paths)  # Images not listed in question_image_mapping.csv

                # Process mapped images
                for image_path in tqdm(image_paths, desc=f"Processing Mapped Images for QID {q_id}", leave=False):
                    try:
                        image = Image.open(image_path)
                    except UnidentifiedImageError as e:
                        tqdm.write(f"Skipping image {image_path} due to unsupported format or corrupt file: {e}")
                        continue

                    image_number = os.path.splitext(os.path.basename(image_path))[0]
                    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
                    if torch.cuda.is_available():
                        inputs = {key: value.cuda() for key, value in inputs.items()}

                    outputs = model(**inputs)
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    similarity = cosine_similarity(text_features, image_features, dim=1).mean().item()

                    csv_writer.writerow([q_id,text, image_path, image_number, similarity, "Mapped"])

                # Process additional images
                for image_path in tqdm(additional_images, desc=f"Processing Unmapped Images for QID {q_id}", leave=False):
                    try:
                        image = Image.open(image_path)
                    except UnidentifiedImageError as e:
                        tqdm.write(f"Skipping image {image_path} due to unsupported format or corrupt file: {e}")
                        continue

                    image_number = os.path.splitext(os.path.basename(image_path))[0]
                    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
                    if torch.cuda.is_available():
                        inputs = {key: value.cuda() for key, value in inputs.items()}

                    outputs = model(**inputs)
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    similarity = cosine_similarity(text_features, image_features, dim=1).mean().item()

                    csv_writer.writerow([q_id, text, image_path, image_number, similarity, "Unmapped"])

print("Cosine similarity results have been saved to 'cosine_similarity_results.csv'")



# import pickle
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image, UnidentifiedImageError
# from torch.nn.functional import cosine_similarity
# import torch
# from tqdm import tqdm 

# # Load the text and image data
# with open('QID_ques.pkl', 'rb') as f:
#     text_data = pickle.load(f)

# with open('question_image_dict.pkl', 'rb') as f:
#     image_data = pickle.load(f)

# # Load the CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Check if CUDA is available and move model to GPU
# if torch.cuda.is_available():
#     model = model.cuda()
#     print("Using CUDA")

# results = {}

# # Loop through each question ID using tqdm for the outer loop
# for q_id in tqdm(text_data.keys(), desc="Processing Questions"):
#     if q_id in image_data and q_id in text_data:
#         text = text_data[q_id].split('[SEP]')
#         image_paths = image_data[q_id]
#         similarity_scores = []

#         # Process each image path for the current question ID
#         for image_path in tqdm(image_paths, desc=f"Processing Images for QID {q_id}", leave=False):
#             try:
#                 image = Image.open(image_path)
#             except UnidentifiedImageError as e:
#                 tqdm.write(f"Skipping image {image_path} due to unsupported format or corrupt file: {e}")
#                 continue  # Skip this image and continue with the next

#             # Process inputs and ensure they are on the same device as the model
#             inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
#             if torch.cuda.is_available():
#                 inputs = {key: value.cuda() for key, value in inputs.items()}  # Move input tensors to GPU

#             outputs = model(**inputs)
#             image_features = outputs.image_embeds
#             text_features = outputs.text_embeds

#             # Compute cosine similarity
#             similarity = cosine_similarity(text_features, image_features, dim=1).item()
#             similarity_scores.append(similarity)

#         results[q_id] = similarity_scores
#     else:
#         tqdm.write(f"Skipping QID {q_id} due to missing text or images.")

# # Save the cosine similarity results
# with open('output_cosine_similarities.pkl', 'wb') as f:
#     pickle.dump(results, f)

# #--------To Preview the above Data-------#

# import pickle
# import pandas as pd

# # Load the question text data
# with open('QID_ques.pkl', 'rb') as f:
#     text_data = pickle.load(f)

# # Load the image paths data
# with open('question_image_dict.pkl', 'rb') as f:
#     image_data = pickle.load(f)

# # Load the cosine similarities data
# with open('output_cosine_similarities.pkl', 'rb') as f:
#     similarity_data = pickle.load(f)

# # Create a DataFrame to store the preview
# preview_data = []

# # Populate the DataFrame
# for q_id, images in image_data.items():
#     question_text = text_data.get(q_id, "Question not found")
#     similarities = similarity_data.get(q_id, [])
#     for img_path, similarity in zip(images, similarities):
#         preview_data.append({
#             "Question ID": q_id,
#             "Question": question_text,
#             "Image Path": img_path,
#             "Cosine Similarity": similarity
#         })

# # Create DataFrame
# df = pd.DataFrame(preview_data)

# # Display the DataFrame
# print(df.head().to_string())
