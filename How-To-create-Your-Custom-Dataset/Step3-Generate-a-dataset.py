import os
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionPipeline

# Define classes
wild_animal_classes = [
    "Lion", "Tiger", "Elephant", "Leopard", "Wolf", 
    "Bear", "Giraffe", "Zebra", "Cheetah", "Hippopotamus"
]

# Define prompts for each class
animal_prompts = {
    "Lion": "A majestic lion standing in the savannah, golden fur, realistic, ultra-detailed, sharp focus, photorealistic",
    "Tiger": "A powerful Bengal tiger walking in a dense jungle, orange fur with black stripes, ultra-detailed, photorealistic",
    "Elephant": "A giant African elephant standing in a grassland, large ears, realistic texture, photorealistic",
    "Leopard": "A spotted leopard climbing a tree in the wild, muscular body, sharp gaze, ultra-detailed, photorealistic",
    "Wolf": "A wild gray wolf howling in a snowy forest, thick fur, sharp eyes, ultra-detailed, photorealistic",
    "Bear": "A massive brown bear standing near a river, wet fur, muscular body, ultra-detailed, photorealistic",
    "Giraffe": "A tall giraffe eating leaves from a tree in the African savannah, long neck, ultra-detailed, photorealistic",
    "Zebra": "A zebra running across the grasslands, black and white stripes, ultra-detailed, photorealistic",
    "Cheetah": "A fast cheetah sprinting through the savannah, spotted fur, muscular legs, ultra-detailed, photorealistic",
    "Hippopotamus": "A large hippopotamus standing in a river, wet skin, powerful body, ultra-detailed, photorealistic"
}

# Negative prompt
negative_prompt = "blurry, distorted, unrealistic, low-quality, bad anatomy, extra limbs, unnatural colors"

# Load the model
model_id = "dreamlike-art/dreamlike-diffusion-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")

# Create dataset folders
output_dir = "d:/temp/wildlife_dataset"
os.makedirs(output_dir, exist_ok=True)

# Generate images
num_images = 100
image_size = 640
params = {
    'num_inference_steps': 100,
    'width': image_size,
    'height': image_size,
    'negative_prompt': negative_prompt
}

for animal in wild_animal_classes:
    class_dir = os.path.join(output_dir, animal)
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(num_images):
        print(f"Generating {animal} image {i+1}/{num_images}")
        img = pipe(animal_prompts[animal], **params).images[0]
        img_array = np.array(img)[:, :, ::-1]  # Convert to BGR for OpenCV
        
        # Save image
        image_path = os.path.join(class_dir, f"{animal}_{i+1}.jpg")
        cv2.imwrite(image_path, img_array)
        
        # Display image
        cv2.imshow(f"{animal} {i+1}/{num_images}", img_array)
        cv2.waitKey(1000)  # Display for 1000ms
        cv2.destroyAllWindows()

print("Dataset generation complete!")
