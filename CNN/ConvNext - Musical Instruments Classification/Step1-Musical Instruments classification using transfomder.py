# Dataset : https://www.kaggle.com/datasets/gpiosenka/musical-instruments-image-classification/data


import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
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


# Print information about the dataset
print("\nDataset information:")
for split_name, split_dataset in dataset.items():
    print(f"\n{split_name} split:")
    print(f"Number of samples: {len(split_dataset)}")
    print(f"Features: {split_dataset.features}")
    if 'label' in split_dataset.features:
        print(f"Labels: {split_dataset.features['label'].names}")

print("=================================================")

# Extract the first image from the train split
first_sample_image = dataset["train"][0]
print("Keys in first sample:", first_sample_image.keys())
# Get the image data from the first sample
first_image = first_sample_image["image"]
first_label = first_sample_image["label"]
print("First image type:", type(first_image))

# Display the image using Matplotlib
plt.imshow(first_image)
plt.axis("off")  # Turn off axis labels for better visualization
plt.title("First Image from Dataset : " + str(first_label))
plt.show()

print("=================================================")

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


print("Label name of the first image : " + id2label[first_label])
print("=================================================")
# =================================================================================

# Process data #
# ============ #

# After we've created a dataset, it's time to prepare it for the model. 
# We first load the image processor corresponding to the pre-trained model which we'll fine-tune.


from transformers import AutoImageProcessor

# Prepare the images for the model
# There are many ConvNext models : https://huggingface.co/models?search=convnext
# We will choose the "convnext-base-224". You can use the "convnext-base-224-22K" as well  

image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k")


print("==============  image_processor :" )
print(image_processor)

# During training, we'll define some image transformations like random flipping/cropping in order to create a robust image classifier 
# (as it still needs to predict the right label no matter the orientation or size of the image). 
# This is also known as "image augmentation" as we're augmenting the dataset with all kinds of transformations.

# Image transformation 


from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

# normaize the images by the image_mean and image_std
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

#The RandomResizedCrop does two important things:
#Resizes images to a consistent size (224 pixels for the shortest edge)
#Provides data augmentation by randomly cropping during training
transform = Compose(
    [
     RandomResizedCrop(image_processor.size["shortest_edge"]),
     RandomHorizontalFlip(),
     ToTensor(), # convert form Pillow to Pytorch Tensor
     normalize # use the normalize we define early 
    ]
)

def train_transforms(examples):
  examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]

  return examples


# Print again the dataset
print ("*********** Dataset : ")
print(dataset)


processed_dataset = dataset.with_transform(train_transforms)

print("*************** processed_dataset[train][0] :")
print(processed_dataset["train"][0])  # .keys())

print("*************** processed_dataset[train][0][pixel_values].shape :")
print(processed_dataset["train"][0]["pixel_values"].shape)

print("*************** processed_dataset[train][0].keys :")
print(processed_dataset["train"][0].keys())


# You can see the values normalize 

# =======================================================================================
# Create PyTorch DataLoader

# Next we create a PyTorch dataloader which allows us to get batches of training images and corresponding labels. 
# This is required as neural networks are typically trained on batches rather than on individual items at a time.

from torch.utils.data import DataLoader

def collate_fn(examples):
  pixel_values = torch.stack([example["pixel_values"] for example in examples])
  labels = torch.tensor([example["label"] for example in examples])

  return {"pixel_values": pixel_values, "labels": labels}


# Create PyTorch DataLoaders for both train and test datasets
train_dataloader = DataLoader(processed_dataset["train"], collate_fn=collate_fn, batch_size=8, shuffle=True)

# when you split the dataset earlier in the code, you used split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42), which created train_dataset and test_dataset as splits of the original "train" key.

val_dataloader = DataLoader(processed_dataset["train"], collate_fn=collate_fn, batch_size=8, shuffle=False)

# Get the first batch of images 
batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)

# ====================================================================================
# Define model

# Next we'll define our pre-trained model which we'll fine-tune. 
# We replace the image classification head on top 
# (which had 1000 classes as it was pre-trained on ImageNet) 
# and replace it by a randomly initialized one with 10 output neurons (as we have 30 classes).


from transformers import AutoModelForImageClassification

# This model is for 1000 Classes (Imagenet) and we need our 30 Classes
# The "ignore_mismatched_sizes" parameter will replace it with our classes 
model = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224-22k",
                                                        id2label=id2label,
                                                        label2id=label2id,
                                                        ignore_mismatched_sizes=True) # this 



# Train the model
# ***************

# Here we'll create a basic PyTorch training loop, 
# which goes over the data multiple times. 
# The parameters of the model are updated using backpropagation + stochastic gradient descent.

from tqdm import tqdm
import os

# Directory to save checkpoints
save_dir = "d:/temp/models/ConvNext-30 Musical Instruments/checkpoints/"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()


# Initialize variables for tracking the best loss and early stopping
best_loss = float("inf")
epochs_without_improvement = 0
patience = 10  # Stop training if no improvement for these many epochs
max_epochs = 100  # Limit to 100 epochs
best_model_path = os.path.join(save_dir, "best_model.pth")

# Training Loop
for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Load best model weights at the start of each epoch
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model weights for fine-tuning.")

    # Training step
    model.train()
    for batch in tqdm(train_dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU

        optimizer.zero_grad()
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss, logits = outputs.loss, outputs.logits
        loss.backward()
        optimizer.step()

        # Metrics
        train_loss += loss.item()
        train_total += batch["labels"].shape[0]
        train_correct += (logits.argmax(-1) == batch["labels"]).sum().item()

    # Calculate training metrics
    train_accuracy = train_correct / train_total
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss, logits = outputs.loss, outputs.logits

            val_loss += loss.item()
            val_total += batch["labels"].shape[0]
            val_correct += (logits.argmax(-1) == batch["labels"]).sum().item()

    # Calculate validation metrics
    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

    # Check for improvement in validation loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        epochs_without_improvement = 0  # Reset patience counter
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with Validation Loss: {best_loss:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement. Patience count: {epochs_without_improvement}/{patience}")

    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement.")
        break