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



SIZE = 224
BATCH_SIZE = 16

df = pd.read_csv("/mnt/d/Data-Sets-Image-Classification/Butterfly Image Classification/Training_set.csv")

# Calulate how many classes we have (exptected 75 classes)
classes_count = df['label'].nunique()
print("Number of classes: " + str(classes_count))


train_df , val_df = train_test_split(df, test_size=0.2, random_state=42) 

image_dir = "/mnt/d/Data-Sets-Image-Classification/Butterfly Image Classification/train"

train_datagen = ImageDataGenerator(rescale=1./255,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Create the CNN model

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(classes_count, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
best_model_file = '/mnt/d/Temp/Models/Best_Butterfly-Image-Classification.keras'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', save_best_only=True, verbose=1)

PATIENCE = 5
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

steps_per_epoch = int(train_generator.samples / BATCH_SIZE)
validation_steps = int(val_generator.samples / BATCH_SIZE)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[best_model, early_stopping] )


# Draw the training and validation accuracy and loss curves

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()


