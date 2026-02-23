import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers


# Load the train data 
df = pd.read_csv("/mnt/d/Data-Sets-Image-Classification/Butterfly Image Classification/Training_set.csv")
print(df.head(10))
print("Number of train images: " + str(len(df)))

# Calulate the number of images per class
class_counts = df['label'].value_counts().sort_index()

plt.figure(figsize=(14,8)) 
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Distribution of Classes in the Butterfly Dataset')
plt.xlabel('Butterfly Classes')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



# Display a random images :

image_dir = "/mnt/d/Data-Sets-Image-Classification/Butterfly Image Classification/train"
sample_images = df.sample(9, random_state=42)

fig,axes = plt.subplots(3,3, figsize=(12,12))

for i , (index , row) in enumerate(sample_images.iterrows()):
    img_path = os.path.join(image_dir, row['filename'])
    img = load_img(img_path, target_size=(150,150))
    img_array = img_to_array(img) / 255.0 # Normalize the image

    ax = axes[i // 3, i % 3]
    ax.imshow(img_array)
    ax.set_title(f"Class: {row['label']}")
    ax.axis('off')

plt.tight_layout()
plt.show()

