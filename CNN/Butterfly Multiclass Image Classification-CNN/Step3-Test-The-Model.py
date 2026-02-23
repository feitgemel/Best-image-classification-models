import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers


SIZE = 224
BATCH_SIZE = 16

df = pd.read_csv("/mnt/d/Data-Sets-Image-Classification/Butterfly Image Classification/Training_set.csv")

# Calulate how many classes we have (exptected 75 classes)
classes_count = df['label'].nunique()
print("Number of classes: " + str(classes_count))


train_df , val_df = train_test_split(df, test_size=0.2, random_state=42) 

image_dir = "/mnt/d/Data-Sets-Image-Classification/Butterfly Image Classification/train"

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# load the saved model
best_model_file = '/mnt/d/Temp/Models/Best_Butterfly-Image-Classification.keras'
model = tf.keras.models.load_model(best_model_file)
print("Model loaded successfully")


val_images, val_labels = next(val_generator)
pred_labels = model.predict(val_images)
pred_labels = np.argmax(pred_labels, axis=1)
true_labels = np.argmax(val_labels, axis=1)

class_indices = val_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}

def display_images(images , true_labels, pred_labels, class_names, num_images=9):
    plt.figure(figsize=(15,15))

    for i in range(num_images):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        true_label_name = class_names[int(true_labels[i])]
        pred_label_name = class_names[pred_labels[i]]
        plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Run the function to display the images with their true and predicted labels
display_images(val_images, true_labels, pred_labels,class_names, num_images=9)

