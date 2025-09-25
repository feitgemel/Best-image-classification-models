import numpy as np
import pandas as pd
import os 
import random
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input 
from tensorflow.keras.optimizers import Adam

base_dir = "/mnt/d/Data-Sets-Image-Classification/Knee Osteoarthritis Dataset with Severity Grading"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

BATCH_SIZE = 16 
IMG_SIZE = (224, 224) # ResNet152V2 expects 224x224 images
IMG_DIM = (224, 224, 3) # 3 channels for RGB
NUM_CLASSES = 5 # 5 classes for Knee Osteoarthritis severity grading (0-4)

# Lets plot some sample images from the dataset

fig , ax = plt.subplots(5, 5, figsize=(18, 18))

for class_id in range(5):
    folder = os.path.join(train_dir, str(class_id))
    os.chdir(folder)
    samples = random.sample(os.listdir(folder), 5) 

    for col in range(5):
        image = cv2.imread(samples[col])
        ax[class_id, col].imshow(image)
        ax[class_id, col].set_title(f"Class: {class_id}")
        ax[class_id, col].axis('off')

plt.show()

# ========================================================================================================

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE) 
    image = preprocess_input(image)
    # convert the label to one-hot encoding
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


# load the datasets 
print("Loading training dataset... ")
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int',
    color_mode='rgb',
    verbose=True ).map(preprocess)

print("Loading validation dataset... ")
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int',
    color_mode='rgb',
    verbose=True ).map(preprocess)

print("Loading test dataset... ")
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int',
    color_mode='rgb',
    verbose=True ).map(preprocess)


# Build the model using ResNet152V2

# Define Data Augmentation :
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),]) 


# Build the model using ResNet152V2 with Augmentation
inputs = Input(shape=IMG_DIM)
x = data_augmentation(inputs) # Apply data augmentation only during training
x = preprocess_input(x) # Preprocess input for ResNet152V2 (Normelization)

# Base model
base_model = ResNet152V2(input_shape=(224,224,3), 
                         include_top=False, 
                         weights='imagenet',
                         pooling='avg') # Global Average Pooling

# Freeze the base model
x = base_model(x, training=False)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

# Output layer
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# Final model
model = Model(inputs, outputs)

# Define the learning rate (low value)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr = 0.00001) 
early_stopping = EarlyStopping(monitor='val_loss', patience=25) 

# save the best model
model_checkpoint = ModelCheckpoint(
    filepath='/mnt/d/temp/models/Knee Osteoarthritis_best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1) 

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=500,
    callbacks=[reduce_lr, early_stopping, model_checkpoint],
    verbose=1) 

# Evaluate the model on the test dataset
model.evaluate(test_dataset)

# Plot the Training and Validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


