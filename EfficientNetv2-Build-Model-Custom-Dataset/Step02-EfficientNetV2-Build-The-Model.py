import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

import numpy as np 
import os
import matplotlib.pyplot as plt 

IMAGE_SIZE=224 # all the images will be resize

# get classes names
imageFolder = 'e:/data-sets/olympics'
CLASSES = os.listdir(imageFolder)
num_classes = len(CLASSES)

print(CLASSES)
print(num_classes)

# load the pre-trained model - EfficientNetV2 model

base_model = tf.keras.applications.EfficientNetV2S(weights='imagenet', input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False)
# include_top=False -> chop the last layer (1000 classes )

# all the layers will be fine tuned during training
base_model.trainable = True

# create a new model with more layers for our data

model = tf.keras.Sequential([ 
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

from tensorflow.keras.optimizers import Adam
adam_opt = Adam(learning_rate = 0.0001) # low value for transfer learning

# compile the model
model.compile(optimizer=adam_opt , loss='categorical_crossentropy', metrics=['accuracy'])

# load the data
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2 ,
    zoom_range=0.2 ,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_folder = 'e:/data-sets/olympics-train'
test_folder = 'e:/data-sets/olympics-test'

train_generator = train_datagen.flow_from_directory(
    train_folder, 
    target_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size=8,
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_folder, 
    target_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size=8,
    class_mode = 'categorical',
    color_mode = 'rgb')

EPOCHS=300
best_model_file = 'e:/temp/olympics-EfficientNetV2.h5'


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy" , patience=25, factor=0.1 , verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy" , patience=25, verbose=1)]


result = model.fit(
    train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=callbacks )

# get the index of the epoch with the highest validation accuracy
best_val_acc_epoch = np.argmax(result.history['val_accuracy'])

# get the best validation accuracy value
best_val_acc = result.history['val_accuracy'][best_val_acc_epoch]

print("Best validation accuracy : " + str(best_val_acc))


# Plot the accuracy :

plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


# Plot the loss 

plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()








