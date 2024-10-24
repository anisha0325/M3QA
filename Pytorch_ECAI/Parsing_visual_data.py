#-------CODE FOR JSON and PICKLE file creation--------#

import re
import pickle
import json
import os
import glob

# Open the text file for reading
input_file = "Dataset/New Text Document.txt"  # Your input text file
image_dir = "Dataset/Annotations/"  # Directory containing images

# List of image file extensions to include
image_extensions = ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.avif', '*.gif', '*.svg', '*.JPG', '*.com', '*.cms']

# Initialize an empty dictionary to store the data
question_image_dict = {}

# Function to find image paths in the given directory and context folder
def find_images(context_id, image_id):
    context_path = os.path.join(image_dir, f"F{context_id}")
    
    # Search for files matching the image_id with any of the possible extensions
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(context_path, f"i{image_id.strip()}{ext[1:]}")))
    
    return image_paths

# Process the input file
with open(input_file, 'r') as infile:
    for line in infile:
        # Use regular expressions to extract the relevant parts
        match = re.match(r"([a-f0-9]+)\s+(\d+)\s+\{([\d,]*)\}", line)
        if match:
            question_id = match.group(1)
            context_id = match.group(2)
            image_ids = match.group(3).split(',')  # Split image_ids by comma

            # Find the image paths for each image_id in the given context
            image_paths = []
            for image_id in image_ids:
                found_paths = find_images(context_id, image_id)
                image_paths.extend(found_paths)  # Add all valid paths

            # Add or update the dictionary for this question_id
            if question_id not in question_image_dict:
                question_image_dict[question_id] = image_paths
            else:
                question_image_dict[question_id].extend(image_paths)  # If already exists, append the new paths

# Print the dictionary
for question_id, image_paths in question_image_dict.items():
    print(f"{question_id}: {image_paths}")

# Save the dictionary as a pickle file
with open('question_image_dict.pkl', 'wb') as pickle_file:
    pickle.dump(question_image_dict, pickle_file)

# Save the dictionary as a JSON file
with open('question_image_dict.json', 'w') as json_file:
    json.dump(question_image_dict, json_file, indent=4)

print("Dictionary saved as both pickle and JSON formats.")


#-----Code to Detect Image Extensions Dynamically:-----#

# import csv
# import os
# import re
# import glob

# # Input and output file paths
# input_file = "/home/ninad/vaibhav_r/MedQA/Dataset_MedQA/Image_Document.txt"  # Your input text file
# output_file = "/home/ninad/vaibhav_r/MedQA/Dataset_MedQA/question_image_dict.csv"  # Output CSV file
# image_dir = "/home/ninad/vaibhav_r/MedQA/Dataset_MedQA/Annotations/"  # Directory containing images

# # List of image file extensions to include
# image_extensions = ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.avif', '*.gif', '*.svg', '*.JPG', '*.com', '*.cms']

# # Function to find image paths in the given directory and context folder
# def find_images(context_id, image_id):
#     context_path = os.path.join(image_dir, f"F{context_id}")
    
#     # Search for files matching the image_id with any of the possible extensions
#     image_paths = []
#     for ext in image_extensions:
#         image_paths.extend(glob.glob(os.path.join(context_path, f"i{image_id.strip()}{ext[1:]}")))
    
#     return image_paths

# # Open the input and output files
# with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
#     # Create a CSV writer
#     writer = csv.writer(outfile)
    
#     # Write the CSV header
#     writer.writerow(['question_id', 'image_path'])
    
#     # Process each line in the input file
#     for line in infile:
#         # Use regular expressions to extract the relevant parts
#         match = re.match(r"([a-f0-9]+)\s+(\d+)\s+\{([\d,]*)\}", line)
#         if match:
#             question_id = match.group(1)
#             context_id = match.group(2)
#             image_ids = match.group(3).split(',')  # Split image_ids by comma

#             # Write each image_id and its corresponding image path to the CSV
#             for image_id in image_ids:
#                 # Find the corresponding image path for the image_id, checking all extensions
#                 image_paths = find_images(context_id, image_id)
                
#                 # Write all found image paths for this image_id
#                 for image_path in image_paths:
#                     writer.writerow([question_id, image_path])

# print(f"Data successfully written to {output_file}")
