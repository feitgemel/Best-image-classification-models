# dataset : https://www.kaggle.com/datasets/pmigdal/alien-vs-predator-images

import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt 

IMAGE_SIZE = [224,224]
trainMyImagesFolder = "E:/Data-sets/alien_vs_predator_thumbnails/data/train"
testMyImagesFolder = "E:/Data-sets/alien_vs_predator_thumbnails/data/validation"

myResnet = ResNet50(input_shape=IMAGE_SIZE + [3] , weights="imagenet", include_top=False )
print(myResnet.summary())

# freeze the weights
for layer in myResnet.layers:
    layer.trainable = False 

Classes = glob('E:/Data-sets/alien_vs_predator_thumbnails/data/train/*')

print(Classes)
numOfClasses = len(Classes)

# build the model
global_avg_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResnet.output)

PlusFlattenLayer = Flatten()(global_avg_pooling_layer)

# add the last layer
predictionLayer = Dense(numOfClasses, activation='softmax')(PlusFlattenLayer)

model = Model(inputs=myResnet.input , outputs=predictionLayer)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

# data augmentation

train_datagen = ImageDataGenerator(rescale = 1./255 ,
                                   shear_range = 0.2 ,
                                   zoom_range = 0.2 ,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255 )

training_set = train_datagen.flow_from_directory(trainMyImagesFolder,
                                                 target_size = (224,224),
                                                 batch_size=32 ,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(testMyImagesFolder,
                                                 target_size = (224,224),
                                                 batch_size=32 ,
                                                 class_mode = 'categorical')

EPOCHS = 200
best_model_file = 'e:/temp/alien-predator-model.h5'


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau , EarlyStopping

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1 , save_best_only=True , monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=10 , factor=0.1 , verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=30 , verbose=1) ]

# train
r = model.fit(
    training_set,
    validation_data = test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=callbacks )

# print the best validation accuracy
best_val_acc = max(r.history['val_accuracy'])
print(f"Best validation Accuracy : {best_val_acc}")


# plot the results / history

plt.plot(r.history['accuracy'], label='Train acc')
plt.plot(r.history['val_accuracy'], label='Val acc')
plt.legend()
plt.show()

plt.plot(r.history['loss'], label='Train loss')
plt.plot(r.history['val_loss'], label='Val loss')
plt.legend()
plt.show()
