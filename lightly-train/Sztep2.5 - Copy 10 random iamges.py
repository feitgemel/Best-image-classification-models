import os
import random
import shutil

# Set source and target folder paths
source_root = "/mnt/d/Data-Sets-Image-Classification/9 dogs Breeds"
target_root = "/mnt/d/Temp/9 dogs breed images-10-random-images"

# Create target root folder if it doesn't exist
os.makedirs(target_root, exist_ok=True)

# Walk through subfolders in source directory
for subfolder in os.listdir(source_root):
    subfolder_path = os.path.join(source_root, subfolder)
    
    if os.path.isdir(subfolder_path):
        # Get all image files in the current subfolder
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        # Randomly select 10 images (or less if not enough)
        selected_images = random.sample(images, min(10, len(images)))
        
        # Prepare the target subfolder path
        target_subfolder = os.path.join(target_root, subfolder)
        os.makedirs(target_subfolder, exist_ok=True)
        
        # Copy selected images
        for image in selected_images:
            src_image_path = os.path.join(subfolder_path, image)
            dst_image_path = os.path.join(target_subfolder, image)
            shutil.copy2(src_image_path, dst_image_path)

print("Image copy completed.")
