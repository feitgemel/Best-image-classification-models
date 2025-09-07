from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch 

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0" 

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")

# Example 1
prompt = "A brave young girl dressed in medieval leather armor stands in an enchanted forest. Sunlight filters through the tall, ancient trees, casting golden beams on her as she holds a longsword. Her determined gaze suggests she is ready for battle. The background is filled with magical creatures peeking from the shadows. The art style is semi-realistic with fantasy elements, detailed textures, and rich colors."
image = pipe(prompt).images[0]

print("[PROMPT]: ", prompt)
plt.imshow(image)
plt.axis("off")
plt.show()

# Example 2 
prompt2 = "A cheerful young girl with braided auburn hair carries a basket of apples in a bustling medieval market. Wooden stalls line the cobblestone streets, selling fresh bread, textiles, and trinkets. Townsfolk chat, and a bard plays a lute nearby. The art style is colorful and slightly stylized, reminiscent of medieval storybook illustrations."

image = pipe(prompt2).images[0]

print("[PROMPT]: ", prompt2)
plt.imshow(image)
plt.axis("off")
plt.show()

# Example 3
prompt3 = "A girl dressed in a dark hooded cloak stands in the ruins of an old cathedral, holding an ancient book of spells. The full moon shines through the broken stained-glass windows, casting colorful reflections on the stone floor. Her eyes glow faintly with magic as she whispers an incantation. The scene has a gothic fantasy style, with deep shadows and glowing magical effects."
image = pipe(prompt3).images[0]
print("[PROMPT]: ", prompt3)
plt.imshow(image)
plt.axis("off")
plt.show()


