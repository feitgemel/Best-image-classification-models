import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Paths and parameters
train_path = '/mnt/d/Data-Sets-Image-Classification/30 Musical Instruments/train/'
valid_path = '/mnt/d/Data-Sets-Image-Classification/30 Musical Instruments/valid/'

BATCH_SIZE = 8
IMG_SIZE = (128,128)
IMG_DIM = (128,128,3)
EPOCHS = 200 

# Display a sample image 

img = load_img(train_path + 'acordian/010.jpg') 
plt.imshow(img)
img = img_to_array(img)
print(img.shape)
plt.show()

# Load the dataset :
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int'
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int'
)

# Normalize the pixel values to [0, 1]
def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


train_dataset = train_dataset.map(normalize)
valid_dataset = valid_dataset.map(normalize)

# Cache and prefetch the datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
valid_dataset = valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# Build the CNN model

def get_cnn_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3 , padding='same', activation='relu', input_shape=IMG_DIM)) 
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(64, kernel_size=3 , padding='same', activation='relu'))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(128, kernel_size=3 , padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = get_cnn_model()
print(model.summary())


# Save the best model during training using ModelCheckpoint
check_point_path = "/mnt/d/models/Best-CNN-Model-30-Musical-Instruments.keras"
checkpoint_callback = ModelCheckpoint(
    filepath=check_point_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Early stopping to prevent overfitting
erly_stopping_callback = EarlyStopping(monitor='val_loss', patience=40, verbose=1) 

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, erly_stopping_callback]
)

# Save the final model
model.save('/mnt/d/models/Final-CNN-Model-30-Musical-Instruments.keras')

# Plot training & validation accuracy and loss values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
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



