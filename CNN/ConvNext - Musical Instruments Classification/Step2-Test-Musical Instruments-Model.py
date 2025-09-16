

import torch
from datasets import load_dataset
#from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import matplotlib.pyplot as plt
from PIL import Image
import os 


# Define the data files dictionary with multiple image formats
data_files = {
    "train": os.path.join("D:/Data-Sets-Image-Classification/30 Musical Instruments/train", "**", "*.*"),
    "validation": os.path.join("D:/Data-Sets-Image-Classification/30 Musical Instruments/valid", "**", "*.*"),  # Changed to 'valid'
    "test": os.path.join("D:/Data-Sets-Image-Classification/30 Musical Instruments/test", "**", "*.*")
}


# Load the dataset
dataset = load_dataset(
    "imagefolder",
    data_files=data_files,
    split={
        "train": "train",
        "validation": "validation",  # The split name remains "validation" as it's a standard term
        "test": "test"
    }
)
print("dataet : ")
print(dataset)

# Display the dataset information

# Get the labels :
labels = dataset["train"].features["label"].names
print("--->>> Labels - list of the 30 classes :")
print(labels)

# Labels are present as integers, but we can turn them into actual class names as follows:
print("=======")

id2label = {k:v for k,v in enumerate(labels)}
print("id2label:")
print(id2label)
print("=======")
label2id = {v:k for k,v in enumerate(labels)}
print("label2id:")
print(label2id)



# Set device (use GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k")

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

# normaize the images by the image_mean and image_std
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# Define the image transformations
transform = Compose(
    [
     RandomResizedCrop(image_processor.size["shortest_edge"]),
     RandomHorizontalFlip(),
     ToTensor(), # convert form Pillow to Pytorch Tensor
     normalize # use the normalize we define early 
    ]
)


from transformers import AutoModelForImageClassification

# This model is for 1000 Classes (Imagenet) and we need our 30 Classes
# The "ignore_mismatched_sizes" parameter will replace it with our classes 
model = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224-22k",
                                                        id2label=id2label,
                                                        label2id=label2id,
                                                        ignore_mismatched_sizes=True) # this 


# Load the saved model 
# Path to the saved checkpoint
checkpoint_path = "D:/Temp/Models/ConvNext-30 Musical Instruments/checkpoints/best_model.pth"

# Load the state dictionary
state_dict = torch.load(checkpoint_path)

# Load the weights into the model
model.load_state_dict(state_dict)

#print(model.eval())
model.to(device)

# Load and preprocess the new image
new_image_path = "D:/Data-Sets-Image-Classification/30 Musical Instruments/test/clarinet/4.jpg"  
image = plt.imread(new_image_path)  # Read image using plt
input_image = transform(Image.fromarray(image).convert("RGB")).unsqueeze(0).to(device)  # Convert and transform image

# Perform inference
with torch.no_grad():
    outputs = model(pixel_values=input_image)
    logits = outputs.logits
    predicted_label_id = logits.argmax(-1).item()
    predicted_label = id2label[predicted_label_id]

# Display the image with the predicted label in the title
plt.imshow(image)
plt.title(f"Predicted Label: {predicted_label}")
plt.axis("off")  # Remove axes for better visualization
plt.show()
