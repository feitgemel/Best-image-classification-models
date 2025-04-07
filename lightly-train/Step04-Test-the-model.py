import torch
from torchvision import models, transforms, datasets
from torch import nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Path to the trained model
best_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/10-dogs-breed/fine-tune/best_fine_tuned_resnet50.pth"
data_path = "/mnt/d/Data-Sets-Image-Classification/10 dogs Breeds"  # Dataset path to retrieve class names

# Load the dataset structure to get class names
dataset = datasets.ImageFolder(root=data_path, transform=transform)
class_names = dataset.classes  # Extract class names from dataset

# Load the trained model
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)  # Move model to GPU
model.eval()  # Set the model to evaluation mode

test_image_path = "Best-image-classification-models/lightly-train/Beagle.jpg"  # Beagle image
#test_image_path = "Best-image-classification-models/lightly-train/Shih_Tzu.jpg"  # Shih_Tzu
#test_image_path = "Best-image-classification-models/lightly-train/Collie.jpg"  # Collie

# Load the image using PIL
image = Image.open(test_image_path).convert("RGB")

# Resize and transform the image
image_resized = image.resize((224, 224))
image_tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension
image_tensor = image_tensor.to(device)  # Move input tensor to GPU

# Perform inference
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

class_name = class_names[predicted_class]
print(f"Predicted class: {class_name}")

# Overlay class name on the image using PIL
draw = ImageDraw.Draw(image_resized)
font = ImageFont.load_default()  # Load default font
draw.text((10, 10), class_name, fill="white", font=font)  # Position and color of the text

# Display the image with Matplotlib
plt.imshow(image_resized)
plt.axis('off')  # Turn off axis
plt.show()