# Dataset : https://www.kaggle.com/datasets/gatewayadam/aerospace-images

# The dataset has a file named rename.py -> You can delete it

import tensorflow as tf 
path = "C:/Data-sets/aerospace_images"

batch_size = 16

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

image_generator = ImageDataGenerator(rescale=1/255. , horizontal_flip=True,
                                     zoom_range=0.2 , validation_split=0.2)

train_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=path,
                                                    shuffle=True,
                                                    target_size=(224,224),
                                                    subset="training", # train
                                                    class_mode="categorical" )

validation_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=path,
                                                    shuffle=True,
                                                    target_size=(224,224),
                                                    subset="validation", # validation
                                                    class_mode="categorical" )

IMG_SHAPE = (224,224,3)
num_of_categories = 7

base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable=False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_of_categories, activation='softmax')])

model._name = "Air_VGG19"

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(
    train_dataset,
    epochs=50,
    validation_data = validation_dataset, verbose=1 )


import matplotlib.pyplot as plt


plt.plot(hist.history["loss"])
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_loss"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Loss", "Accuracy", "Validation loss", "Validation Accurcay"])
plt.show()

model.save("e:/temp/air-vgg19.h5")



