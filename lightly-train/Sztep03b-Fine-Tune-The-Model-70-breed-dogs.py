import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Paths for the datasets
#data_set_path = "/mnt/d/Data-Sets-Image-Classification/9 dogs Breeds"
data_set_path = "/mnt/d/Temp/9 dogs breed images-10-random-images"

# Paths for the exported models
last_model = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/70-dogs-Breed/exported_models/exported_last.pt"

# Paths for the first model
#best_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/9-dogs-Breed/fine-tune/best_fine_tuned_resnet50.pth"
best_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/9-dogs-breed/fine-tune/best_fine_tuned_resnet50.pth"

#final_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/9-dogs-Breed/fine-tune/final_fine_tuned_resnet50.pth"
final_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/9-dogs-breed/fine-tune/final_fine_tuned_resnet50.pth"


# Ensure the directory exists before saving the model
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
os.makedirs(os.path.dirname(last_model), exist_ok=True)


dataset = datasets.ImageFolder(root=data_set_path, transform=transform) #data_set_path1 or data_set_path2
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# Load the exported model
model = models.resnet50()
model.load_state_dict(torch.load(last_model, weights_only=True, map_location=device))

# Update the classification head with the correct number of classes
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# Move model to GPU
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting fine-tuning...")
num_epochs = 200  
best_loss = float("inf")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    # Wrap dataloader with tqdm for progress tracking
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for inputs, labels in pbar:
            # Move input tensors to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            pbar.set_postfix(loss=loss.item())  # Show loss in tqdm progress bar

    epoch_loss /= len(dataloader)  # Average loss per batch
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}")

    # Save model only if the loss improves
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at epoch {epoch+1} with loss {epoch_loss:.4f}")
        no_improve_count = 0  # Reset counter when there's improvement
    else:
        no_improve_count += 1
        print(f"No improvement at epoch {epoch+1}. Best loss remains {best_loss:.4f}")
        
 
# Save the final trained model
torch.save(model.state_dict(), final_model_path)
print("Final model saved after training.")