from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")

# function for generating images with different parameters
def generate_image(pipe , prompt , params):
    img = pipe(prompt, **params).images  

    num_images = len(img)
    if num_images > 1:
        fig, ax = plt.subplots(nrows=1, ncols=num_images) 
        for i in range(num_images):
            ax[i].imshow(img[i])
            ax[i].axis("off")

    else:
        fig = plt.figure()
        plt.imshow(img[0])
        plt.axis("off")

    plt.tight_layout()


# Run wi no parameters:

prompt = "portrait of a pretty blonde girl, a flower crown, flowing maxi dress with colorful patterns and fringe, a sunset or nature scene, green and gold color scheme"

params = {} 

print("No prameters")
generate_image(pipe, prompt, params)
plt.show()

# ===============================================================================
# Define number inference steps 
params = {"num_inference_steps": 100}

print(params)
generate_image(pipe, prompt, params)
plt.show()

# ===============================================================================
# Define dimensions of the image
params = {"num_inference_steps": 100, "height": int(640 * 1.5), "width": 512 }
print(params)
generate_image(pipe, prompt, params)
plt.show()

# ===============================================================================
# Define Number of images to generate
params = {"num_inference_steps": 100, "num_images_per_prompt": 3}
print(params)
generate_image(pipe, prompt, params)
plt.show()

# ===============================================================================
# Define Negative prompt
params = {"num_inference_steps": 100, "num_images_per_prompt": 3, 
          'negative_prompt': 'ugly, distorted, low quality'}
print(params)
generate_image(pipe, prompt, params)
plt.show()


