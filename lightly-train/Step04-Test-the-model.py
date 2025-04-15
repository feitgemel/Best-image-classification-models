import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os 
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Path to the trained model

best_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/10-dogs-breed/fine-tune/best_fine_tuned_resnet50.pth"
dataset_path = "/mnt/d/Data-Sets-Image-Classification/10 dogs Breeds" # path to the dataset for categories names / lables

# Load the dataset structure to get the class names
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = dataset.classes
print("Class names:", class_names)

model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
model.eval()

#test_image_path = "Best-image-classification-models/lightly-train/Beagle.jpg"
#test_image_path = "Best-image-classification-models/lightly-train/Shih_Tzu.jpg"
test_image_path = "Best-image-classification-models/lightly-train/Collie.jpg"


# load the image using PIL
image = Image.open(test_image_path).convert("RGB")

# Resize and transform the image
image_resized = image.resize((224, 224))
image_tensor = transform(image_resized).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

class_name = class_names[predicted_class]
print("===================================")
print(f"Predicted class: {class_name}")

# Overlay class name on the image using PIL
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
draw.text((10,10), class_name, fill="white", font=font)

# Display the image using matplotlib
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.title(f"Predicted class: {class_name}")
plt.show()







