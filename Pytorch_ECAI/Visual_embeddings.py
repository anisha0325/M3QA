import torch, warnings
from torch import nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import os
from tqdm import tqdm
import pandas as pd
import pickle
import pillow_avif

warnings.filterwarnings("ignore", message="xFormers is not available")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained ViT-B/16 model with DINO weights from torch hub
def load_vit_dino():
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14',pretrained=True)
    model.head = nn.Identity()  # Remove the classification head
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Function to process the image and get embeddings
def get_image_embedding(model, image_path):
    # Define the transformation for the image (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to the input size of ViT-L/14
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )   # Normalize with ImageNet mean and std
    ])

    try:
        # Load image from the path
        img = Image.open(image_path)
        # If the image is palette-based (like GIF/PNG) and has transparency, convert it to RGBA
        if img.mode == "P" or img.mode == "LA" or (img.mode == "RGBA" and "transparency" in img.info):
            img = img.convert("RGBA")
        img = img.convert('RGB')  # Ensure 3-channel image (RGB)
        img_t = transform(img)  # Apply transformations
        img_t = img_t.unsqueeze(0)  # Add batch dimension

        # Get the embedding from the ViT model
        with torch.no_grad():
            embedding = model(img_t.to(device))

        return embedding

    except UnidentifiedImageError:
        print(f"Unsupported or invalid image format: {image_path}")
        return torch.zeros(1, 1024).to(device)

# Load the image paths from your CSV file
def load_image_paths_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    question_image_dict = {}
    
    # Group image paths by question_id
    for _, row in data.iterrows():
        question_id = row['question_id']
        image_path = row['image_path']

        if question_id not in question_image_dict:
            question_image_dict[question_id] = []
        question_image_dict[question_id].append(image_path)
    
    return question_image_dict

# Main function to extract embeddings for all images with tqdm progress bar
def extract_embeddings(csv_file):
    # Load the pre-trained ViT model with DINO weights
    model = load_vit_dino()

    # Load image paths from the CSV file
    question_image_dict = load_image_paths_from_csv(csv_file)

    # Dictionary to store embeddings
    image_embeddings = {}

    # Process each question_id and corresponding image paths
    for question_id, image_paths in tqdm(question_image_dict.items(), desc="Processing question IDs", leave=True):
        image_embeddings[question_id] = []
        for image_path in tqdm(image_paths, desc=f"Processing images for question {question_id}", leave=False):
            if os.path.exists(image_path):
                embedding = get_image_embedding(model, image_path)
                if embedding is not None:
                    image_embeddings[question_id].append(embedding)

    return image_embeddings

# Save the embeddings to a pickle file
def save_embeddings(embeddings, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

# Example usage
if __name__ == "__main__":
    csv_file = "Dataset/question_image_dict.csv"  # Path to your CSV file
    output_file = "full_image_embeddings.pkl"  # Path to save the image embeddings

    # Extract embeddings
    embeddings = extract_embeddings(csv_file)

    # Save the embeddings
    save_embeddings(embeddings, output_file)

    print("Embeddings successfully extracted and saved.")
