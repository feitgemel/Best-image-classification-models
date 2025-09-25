import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the data files dictionary with multiple image formats
data_files = {
    "train": os.path.join("D:/Data-Sets-Image-Classification/30 Musical Instruments/train", "**", "*.*"),
    "validation": os.path.join("D:/Data-Sets-Image-Classification/30 Musical Instruments/valid", "**", "*.*"),
    "test": os.path.join("D:/Data-Sets-Image-Classification/30 Musical Instruments/test", "**", "*.*")
}

# Load the dataset
dataset = load_dataset(
    "imagefolder",
    data_files=data_files,
    split={
        "train": "train",
        "validation": "validation",
        "test": "test"
    }
)

# Get the labels
labels = dataset["train"].features["label"].names
id2label = {k: v for k, v in enumerate(labels)}
label2id = {v: k for k, v in enumerate(labels)}

# Set device (use GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the image processor and define transformations
image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k")
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
transform = Compose([
    RandomResizedCrop(image_processor.size["shortest_edge"]),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize
])

# Load the pre-trained model and weights
model = AutoModelForImageClassification.from_pretrained(
    "facebook/convnext-base-224-22k",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
checkpoint_path = "D:/Temp/Models/ConvNext-30 Musical Instruments/checkpoints/best_model.pth"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Initialize variables for confusion matrix
true_labels = []
predicted_labels = []

# Process all test images
for item in dataset["test"]:
    image_path = item["image"].filename
    true_label = item["label"]
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(pixel_values=input_image)
        logits = outputs.logits
        predicted_label_id = logits.argmax(-1).item()

    # Store labels for confusion matrix
    true_labels.append(true_label)
    predicted_labels.append(predicted_label_id)

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(labels))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot confusion matrix with a larger figure size
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
plt.title("Confusion Matrix")
plt.show()
