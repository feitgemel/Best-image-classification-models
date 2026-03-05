import os 
from diffusers import DiffusionPipeline
import torch
import cv2
import numpy as np

# Link to animals : https://huggingface.co/models?library=diffusers&sort=trending&search=animals

# Configuration 
animals_models = "VuDucQuang/nature-and-animals"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup pipline
pipline = DiffusionPipeline.from_pretrained(
    animals_models, torch_dtype=torch.float16 ) 

pipline.to(device)

object_name = "African elephant"

prompt = f'Medium-shot of a {object_name}, front view, color photography, photorealistic, hyperrealistic, realistic, incredibly detailed, digital art, crisp focus, depth of field, 50mm, 8k'
negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, bad anatomy, blurred, watermark, grainy, signature'

# Generate image
print("Generating image...")
result = pipline(prompt=prompt, negative_prompt=negative_prompt).images[0]


# Convert PIL image to OpenCV format
img = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

# Fixpath to save the image
output_dir = "/mnt/d/temp"

file_path = os.path.join(output_dir, f"{object_name}.png")
cv2.imwrite(file_path, img)
print(f"Image saved to: {file_path}")

# Display the image using OpenCV
cv2.imshow("Generated Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



