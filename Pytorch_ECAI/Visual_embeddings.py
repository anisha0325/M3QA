import torch
from torch import nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import os
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained ViT-B/16 model with DINO weights from torch hub
def load_vit_dino():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
    model.head = nn.Identity()  # Remove the classification head
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Function to process the image and get embeddings
def get_image_embedding(model, image_path):
    # Define the transformation for the image (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to the input size of ViT-B/16
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )   # Normalize with ImageNet mean and std
    ])

    try:
        # Load image from the path
        img = Image.open(image_path).convert('RGB')  # Ensure 3-channel image (RGB)
        img_t = transform(img)  # Apply transformations
        img_t = img_t.unsqueeze(0)  # Add batch dimension

        # Get the embedding from the ViT model
        with torch.no_grad():
            embedding = model(img_t.to(device))

        return embedding

    except UnidentifiedImageError:
        print(f"Unsupported or invalid image format: {image_path}")
        return None

# Load the image paths from your Pickle file
def load_image_paths(pickle_file):
    with open(pickle_file, 'rb') as f:
        question_image_dict = pickle.load(f)
    return question_image_dict

# Main function to extract embeddings for all images
def extract_embeddings(pickle_file):
    # Load the pre-trained ViT model with DINO weights
    model = load_vit_dino()

    # Load image paths from the Pickle file
    question_image_dict = load_image_paths(pickle_file)

    # Dictionary to store embeddings
    image_embeddings = {}

    # Process each question_id and corresponding image paths
    for question_id, image_paths in question_image_dict.items():
        image_embeddings[question_id] = []
        for image_path in image_paths:
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
    pickle_file = "question_image_dict.pkl"  # Path to your pickle file with image paths
    output_file = "image_embeddings.pkl"     # Path to save the image embeddings

    # Extract embeddings
    embeddings = extract_embeddings(pickle_file)

    # Save the embeddings
    save_embeddings(embeddings, output_file)

    print("Embeddings successfully extracted and saved.")
