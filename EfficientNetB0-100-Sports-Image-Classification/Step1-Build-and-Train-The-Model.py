# Download dataset : https://www.kaggle.com/datasets/gpiosenka/sports-classification

import numpy as np
import os 
import random
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers , models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.optimizers import Adam

train_dir = '/mnt/d/Data-Sets-Image-Classification/100 Sports Image Classification/train'
val_dir = '/mnt/d/Data-Sets-Image-Classification/100 Sports Image Classification/valid'
test_dir = '/mnt/d/Data-Sets-Image-Classification/100 Sports Image Classification/test'


# Display a random images with labels from the training dataset

def show_random_images_wth_labels(main_folder , num_images=5):

    subfolders = [os.path.join(main_folder, f ) for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    random.shuffle(subfolders) # Shuffle the list of subfolders to ensure randomness

    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
    for i , subfolder in enumerate(subfolders[:num_images]):
        image_paths = [os.path.join(subfolder, img) for img in os.listdir(subfolder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_paths) # Shuffle the list of image paths to ensure randomness
        image_path = image_paths[0] # Select the first image after shuffling

        img = Image.open(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(os.path.basename(image_path) + f" ({os.path.basename(subfolder)})", fontsize=10)

    plt.tight_layout()
    plt.show()

# Run the function to display random images with labels from the training dataset
show_random_images_wth_labels(train_dir , num_images=5)


# Create an ImageDataGenerator for data augmentation and rescaling (Train / test / validation)

train_dataget = ImageDataGenerator(zoom_range=0.2 , width_shift_range=0.2, height_shift_range=0.2) 

print("Load the train dataset")
train_dg = train_dataget.flow_from_directory(
    train_dir,
    class_mode='categorical',
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    seed=42) 

val_datagen = ImageDataGenerator()

print("Load the Validation dataset")
validation_dg = val_datagen.flow_from_directory(
    val_dir,
    class_mode='categorical',
    target_size=(224,224),
    batch_size=32,
    shuffle=False,
    seed=42)

print("Load the Test dataset")
testing_dg = val_datagen.flow_from_directory(
    test_dir,
    class_mode='categorical',
    target_size=(224,224),
    batch_size=32,
    shuffle=False,
    seed=42)


# Build the model based on EfficientNetB0

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense , Dropout 
from tensorflow.keras.models import Model

# load the EfficientNetB0 model without the top layer and with pre-trained ImageNet weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False


# Add custom top layers for our specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.25)(x)  # Add dropout for regularization
predictions = Dense(100, activation='softmax')(x)  # 100 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)


opt = Adam(learning_rate=0.005)
model.compile(optimizer=opt, loss = keras.losses.categorical_crossentropy , metrics=['accuracy'])

best_model_path = "/mnt/d/temp/models/100-Sports-Model.keras"

history = model.fit(
    train_dg,
    epochs = 50,
    validation_data = validation_dg,
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,mode='min') ,
        ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1, save_best_only=True) ] 
)

# Plotting training and validation accuracy and loss

# extract accuracy and loss values from the history object
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

# plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()  









































































