# Dataset : https://www.kaggle.com/datasets/vinjamuripavan/bird-species

import os 
pathToDataset = "E:/Data-sets/BIRDS 525 SPECIES- IMAGE CLASSIFICATION"
trainPath = pathToDataset + "/train"
testPath = pathToDataset + "/test"
validPath = pathToDataset + "/valid"

no_of_classes = len(os.listdir(trainPath))
print("No. of Classes : " + str(no_of_classes)) 

import matplotlib.pyplot as plt 
import matplotlib.image as mping 
import random 

# view a random image 
def view_random_image(target_dir , target_class) :

    target_folder = target_dir + "/" + target_class 

    # get the random image 
    random_image = random.sample(os.listdir(target_folder), 1)

    # show the image
    img = mping.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    plt.show()

    print(f"Image shape: {img.shape}")

    return img


#img = view_random_image(target_dir=trainPath , target_class="VICTORIA CROWNED PIGEON")


# Build the model 

import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#rescale the image -> from 0-255 to 0-1
train_datagen = ImageDataGenerator(rescale = 1./255) 
test_datagen = ImageDataGenerator(rescale = 1./255) 
valid_datagen = ImageDataGenerator(rescale = 1./255)

train_data = train_datagen.flow_from_directory(directory=trainPath,
                                               batch_size = 32 ,
                                               target_size = (224,224),
                                               class_mode ="categorical")

test_data = test_datagen.flow_from_directory(directory=testPath,
                                               batch_size = 32 ,
                                               target_size = (224,224),
                                               class_mode ="categorical")
valid_data = valid_datagen.flow_from_directory(directory=validPath,
                                               batch_size = 32 ,
                                               target_size = (224,224),
                                               class_mode ="categorical")



# create the model 

base_model = tf.keras.applications.InceptionV3(include_top=False)

# freeze the weights of the model
base_model.trainable = False

# Create inputs into models
inputs = tf.keras.layers.Input( shape=(224,224,3), name = "input-layer")

# pass the inputs
x = base_model(inputs)
print(f"The model shape after passing the inputs : {x.shape}")

# Avegrage pool layer the outputs of the base model
x = tf.keras.layers.GlobalAveragePooling2D(name = "Global-average-pooling-layer")(x)
print(f"The shape after GlobalAveragePoolid2D: {x.shape}")

# create the last output layer
outputs = tf.keras.layers.Dense(no_of_classes, activation='softmax', name='output-layer')(x)

# MMerge the input and the outpus into one model
model = tf.keras.Model(inputs, outputs)

model.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(learning_rate= 0.01),
              metrics = ["accuracy"])

print(model.summary())

EPOCHS = 30
best_model_file = "e:/temp/525-birds.h5"

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True , monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", patience=4, factor=0.1 , verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=5, verbose=1) ]


# train the model 
history = model.fit(train_data,
                    epochs=EPOCHS,
                    steps_per_epoch= len(train_data),
                    validation_data=valid_data,
                    validation_steps= int(0.25 * len(valid_data)),
                    callbacks=callbacks)


# evaluate the test data

print(model.evaluate(test_data))

# plot the results

def plot_loss_curves(history):

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # plot the loss

    plt.plot(epochs, loss, label = "training loss")
    plt.plot(epochs, val_loss, label = "val loss")
    plt.title("loss")
    plt.xlabel(epochs)
    plt.legend()
    plt.show()

    # plot the accuracy 
    plt.plot(epochs, accuracy, label = "training accuracy")
    plt.plot(epochs, val_accuracy, label = "val accuracy")
    plt.title("accuracy")
    plt.xlabel(epochs)
    plt.legend()
    plt.show()


# run the function
plot_loss_curves(history)
