import torch
from transformers import BlipForQuestionAnswering, BlipProcessor
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image

# Resize all images to a consistent size (e.g., 224x224)
def preprocess_images(image_paths, target_size=(224, 224)):
    resized_images = []
    for img_path in image_paths:
        with Image.open(img_path) as img:
            resized_images.append(img.convert("RGB").resize(target_size))
    return resized_images

# Load BLIP-2 model and processor
model_name = "Salesforce/blip-2-flan-t5-xl"  # Replace with appropriate BLIP-2 model path
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
# processor = BlipProcessor.from_pretrained(model_name)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

# Input data
context = "Migraine with brainstem aura or MBA (formerly known as basilar migraines) are headaches that start in the lower part of the brain, called the brainstem. They cause symptoms such as dizziness, double vision, and lack of coordination. These changes, called an aura, can happen about 10 minutes to 45 minutes before your head hurts. The headache pain of a basilar migraine often starts on one side of the head and then gradually spreads and gets stronger. This type of migraine can last anywhere from 4 to 72 hours. And it takes time to recover from one. You may feel drained for up to 24 hours after it's over. Migrainad741b99c42db5747bdb6c205f763e6aes with brainstem aura are known by several different names: Basilar migraine Basilar artery migraine Basilar-type migraine Bickerstaff's syndrome Brainstem migraine Vertebrobasilar migraine MBA can affect people of all ages. Generally, though, they start in childhood or the teen years. Women are slightly more likely to have them than men. Triggers may include: Alcohol Stress Lack of sleep Some medications Hunger Female hormone changes Bright lights Caffeine Nitrites in some foods, like sandwich meat, bacon, and processed foods Overdoing physical activity Weather or altitude Symptoms differ for everyone, but some are typical: Nausea Vomiting Sensitivity to light and sound Cold hands or feet Dizziness Double vision or graying of vision Slurred speech or trouble speaking Temporary blindness Loss of balance Confusion Trouble hearing Body tingling Loss of consciousness Aura symptoms may last between 5 minutes and 1 hour. When the headache starts, you might feel an intense throbbing or pulsating pain on one or both sides of your head or sometimes at the back of your head. After you've had at least two attacks of at least two auras, your doctor often make the diagnosis of migraine with brainstem aura. The condition has many of the same symptoms as another type, called hemiplegic migraine. But the hemiplegic kind usually causes weakness of one side of the body or trouble speaking. Symptoms of MBA can also seem like the signs of other more serious conditions, like seizure disorders, stroke, meningitis, or brain tumors. To rule those out, you'll need to see a brain doctor, called a neurologist. He'll give you a thorough exam and ask you questions about your symptoms. He'll may also use tests like MRI, CT scans, and nerve tests to see what's causing your symptoms. Treatments for MBA generally aim to relieve symptoms of pain and nausea. You might take pain relievers such as acetaminophen, ibuprofen, and naproxen, as well as nausea medicines such as chlorpromazine, metoclopramide, and prochlorperazine. Your doctor may prescribe a medication that treats regular migraines, such as triptans. To keep from getting a MBA, it helps to avoid the things that usually cause one. Keep a journal of your attacks so you can figure out the things that trigger them. It also helps to live a healthy lifestyle. That means you need to: Get enough sleep. Limit your stress. Exercise daily. Diet can also affect migraines. Do these things: Eat a balanced diet. Avoid drugs and alcohol. Don't skip meals. Limit caffeine. Avoid any foods that have been triggers. Some common food triggers include: Dairy Wheat Chocolate Eggs Rye Tomatoes Oranges Along with these changes, If you don't respond to other treatments and you have 4 or more migraine days a month, your doctor may suggest preventive medicines. You can take these regularly to reduce the severity or frequency of the headaches. These include seizure medicines, blood pressure medicines (like beta blockers and calcium channel blockers), and some antidepressants. CGRP inhibitors are a new class of preventive medicine that your doctor may recommend if other medicines don't help."
query = "What causes a migraine with brainstem aura? [SEP] Cause [SEP] What"
instruction = "You are given a set of context sentences and images. Your task is to identify which sentences in the context are most relevant to the question based on both textual and visual information.\n"
images = ["/nethome/asaha/misc/MedQA/Pytorch_ECAI/Dataset/Annotations/F3/i1.png",
 "/nethome/asaha/misc/MedQA/Pytorch_ECAI/Dataset/Annotations/F3/i4.jpg",
 "/nethome/asaha/misc/MedQA/Pytorch_ECAI/Dataset/Annotations/F3/i11.jpg"]  # Replace with paths to your image files
preprocessed_images = preprocess_images(images)


# # Verify image sizes
# for img_path in images:
#     img = Image.open(img_path)
#     print(f"Image {img_path}: size={img.size}, mode={img.mode}")

# breakpoint()

# Combine instruction, query, and context
prompt = (
    f"{instruction}\n\n"
    f"Context: {context}\n"
    f"Question: {query}\n"
    f"List the sentences that are most relevant to answering the question.\n"
)

# Preprocessing inputs
inputs = processor(images=preprocessed_images, text=prompt, return_tensors="pt").to(device)

# Generate predictions
# Generate predictions
outputs = model.generate(
    pixel_values=inputs['pixel_values'],
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=512,
    num_beams=5
)
predicted_answer = processor.decode(outputs[0], skip_special_tokens=True)

# Extract relevant sentences
relevant_sentences = [sentence for sentence in context.split('. ') if sentence in predicted_answer]

# Evaluation metrics
def calculate_metrics(predictions, references):
    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="weighted")
    bleu = sentence_bleu([references], predictions)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True).score(references, predictions)
    meteor = meteor_score([references], predictions)

    return {"accuracy": accuracy, "f1": f1, "bleu": bleu, "rouge": rouge, "meteor": meteor}

# Dummy ground truth for evaluation (Replace with actual references)
ground_truth = ["Dogs are loyal animals."]

metrics = calculate_metrics(relevant_sentences, ground_truth)

# Print outputs
print("Predicted Answer:", predicted_answer)
print("Relevant Sentences:", relevant_sentences)
print("Evaluation Metrics:", metrics)
