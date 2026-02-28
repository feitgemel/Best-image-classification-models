import os 
import pandas as pd
import numpy as np
import keras 
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

path = "/mnt/d/Data-Sets-Image-Classification/Augmented Alzheimer MRI Dataset/AugmentedAlzheimerDataset"

MildDemented_dir = os.path.join(path, "MildDemented")
ModerateDemented_dir = os.path.join(path, "ModerateDemented")
NonDemented_dir = os.path.join(path, "NonDemented")
VeryMildDemented_dir = os.path.join(path, "VeryMildDemented")

# Load data info : 

filepaths = []
labels = []
dict_list = [MildDemented_dir, ModerateDemented_dir, NonDemented_dir, VeryMildDemented_dir]
class_labels = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

for i, j in enumerate(dict_list):
    flist = os.listdir(j)
    for f in flist:
        fpath = os.path.join(j, f)
        filepaths.append(fpath)
        labels.append(class_labels[i])

Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
Alzheimer_data = pd.concat([Fseries, Lseries], axis=1)
Alzheimer_df = pd.DataFrame(Alzheimer_data)

print(Alzheimer_df.head())
print(Alzheimer_df["labels"].value_counts())

# --------------------------------------------------

rest_of_dataset , test_images = train_test_split(Alzheimer_df, test_size=0.3, random_state=42) 
train_set , val_set = train_test_split(rest_of_dataset, test_size=0.2, random_state=42)

print("Train , test and validation set shapes : ", train_set.shape, test_images.shape, val_set.shape)

# --------------------------------------------------

Size = 299 # Xception model input size 299X299 pixels
batch_size = 8

image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)

train = image_gen.flow_from_dataframe(dataframe = train_set,
                                      x_col = "filepaths",
                                      y_col = "labels",
                                      target_size = (Size, Size),
                                      class_mode = "categorical",
                                      batch_size = batch_size,
                                      color_mode = "rgb",
                                      shuffle = False)

test = image_gen.flow_from_dataframe(dataframe = test_images,
                                     x_col = "filepaths",
                                     y_col = "labels",
                                     target_size = (Size, Size),
                                     class_mode = "categorical",
                                     batch_size = batch_size,
                                     color_mode = "rgb",
                                     shuffle = False)


val = image_gen.flow_from_dataframe(dataframe = val_set,
                                     x_col = "filepaths",
                                     y_col = "labels",
                                     target_size = (Size, Size),
                                     class_mode = "categorical",
                                     batch_size = batch_size,
                                     color_mode = "rgb",
                                     shuffle = False)

# Extract the classes 
print("Calculate the number of classes") 
Classes = list(train.class_indices.keys())
print("Classes : ", Classes)
no_of_classes = len(Classes)


# Display images from the train set 
def show_images(image_gen):
    dict_classes = train.class_indices
    classes = list(dict_classes.keys())
    images , labels = next(image_gen) # Get a sample of images and labels from the generator
    plt.figure(figsize=(20,20)) 
    length = len(labels)
    if length < 25 :
        r = length
    else :
        r = 25

    for i in range(r):
        plt.subplot(5,5,i+1)
        image = (images[i]+ 1)/2 # Rescale the image to [0,1] range for visualization
        plt.imshow(image)
        index = np.argmax(labels[i]) # Get the index of the class label
        class_name = classes[index] # Get the class name using the index
        plt.title(class_name, color="green", fontsize=16)
        plt.axis("off")
    plt.show()



# Run the function to show images from the train set
show_images(train)

# Build the Xception model :
 
from tensorflow.keras.optimizers import Adamax 

img_shape = (Size, Size, 3)
base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False, input_shape=img_shape)

# Create the model using the functional API
inputs = tf.keras.Input(shape=img_shape)
x = base_model(inputs) 
x = Flatten()(x)
x = Dropout(rate=0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(rate=0.25)(x)
outputs = Dense(no_of_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adamax(learning_rate=0.001), loss = "categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

# Train the model

history = model.fit(train, epochs=10, validation_data=val, validation_freq=1)

model.save("/mnt/d/temp/models/Alzheimer_Model.keras")


# Check the results on the test set :

pred = model.predict(test)
pred = np.argmax(pred, axis=1) # pick class with highest probability

labels = (train.class_indices)
labels = dict((v,k) for k,v in labels.items()) # reverese the dictionary to get class names from indices
pred2 = [labels[k] for k in pred] # get class names from predicted indices


# plot the results
plt.plot(history.history["accuracy"]) 
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])   
plt.title("Model Loss") 
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()


# Geberate confusion matrix 
from sklearn.metrics import confusion_matrix, accuracy_score

y_test = test_images.labels # set y_test to the expected labels from the test set
print(classification_report(y_test, pred2)) # print the classification report
print("Accuracy of the model on the test set : ","{:.1f}%".format(accuracy_score(y_test, pred2)*100)) # print the accuracy of the model on the test set

# Define the class labels 
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']

# Calculate the confusion matrix
cm = confusion_matrix(y_test, pred2)

# create a heatmap of the confusion matrix
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", vmin=0)

# Set tick labels and axis labels
plt.xticks(ticks = [0.5, 1.5, 2.5, 3.5], labels=class_labels) 
plt.yticks(ticks = [0.5, 1.5, 2.5, 3.5], labels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Set the title
plt.title("Confusion Matrix")

# show the result of the confusion matrix
plt.show()























































































