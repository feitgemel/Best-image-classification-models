import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Path to dataset 
dataset_path = "/mnt/d/Data-Sets-Image-Classification/10 dogs Breeds"

#Path to exported model
last_model = "/mnt/d/temp/models/lightly-train/Object-Classification/out/9-dogs-Breed/exported_models/exported_last.pt"

# Path to output best model
best_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/10-dogs-breed/fine-tune/best_fine_tuned_resnet50.pth"

# path to the final model
final_model_path = "/mnt/d/Temp/Models/lightly-train/Object-Classification/out/10-dogs-breed/fine-tune/final_fine_tuned_resnet50.pth"

# Make sure the direcroty exists beofre saving the models
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
os.makedirs(os.path.dirname(last_model), exist_ok=True)

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# load the pre-trained model
model = models.resnet50()
model.load_state_dict(torch.load(last_model, weights_only=True, map_location=device))
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# move the model to the device
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting fine-tuning...")
num_epochs = 200
patience = 10
best_loss = float('inf')
no_improvement_count = 0

for epoch in range(num_epochs):
    epoch_loss = 0.0

    # Wrap dataloader with tqdm for progress bar
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar :
        for inputs , labels in pbar:

            #Move input tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            pbar.set_postfix(loss=loss.item()) # Show loss in the tqdm progress bar

    epoch_loss /= len(dataloader) 
    print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {epoch_loss:.4f}")

    # Save model only if the loss improves :
    if epoch_loss < best_loss :
        best_loss = epoch_loss 
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at epoch {epoch+1} with loss {best_loss:.4f}")
        no_improvement_count = 0 # Reset the counter
    else:
        no_improvement_count += 1
        print(f"No improvemnet in epoch {epoch+1}. Best loss so far: {best_loss:.4f}")
        print(f"Epochs without improvement: {no_improvement_count}/{patience}")

        # check for early stopping
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs due to no improvement for {patience}. consecutive epochs")
            break

# Save the final model
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")


