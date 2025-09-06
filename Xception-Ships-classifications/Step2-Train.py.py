# Download dataset : https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets/data

import cv2 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import numpy as np

# Load the daat from the csv file
path_to_train_csv = '/mnt/d/Data-Sets-Image-Classification/Ships-dataset/train/train.csv'

data_csv = pd.read_csv(path_to_train_csv)
print("data shape: ", data_csv.shape)

categories = {0: 'Cargo' , 1: 'Military', 2: 'Carrier', 3: 'Cruise', 4: 'Tankers'}

data_csv['category'] = data_csv['category'] - 1 
data_csv['label'] = data_csv['category'].map(categories)
data_csv['label'] = pd.Categorical(data_csv['label'])

# ============================================================

# Load the data 
X = np.load("/mnt/d/temp/data/ships_data.npy")
print("X shape: ", X.shape)

# ===========================================================

from sklearn.preprocessing import OneHotEncoder
y = OneHotEncoder(dtype='int8', sparse_output=False).fit_transform(data_csv['category'].values.reshape(-1, 1))
print("y shape: ", y.shape)

from sklearn.model_selection import train_test_split

X_data , X_test, y_data, y_test = train_test_split(X, y, test_size=0.15 , random_state=42) 
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2 , random_state=42)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(horizontal_flip=True,
                               rotation_range=45,
                               zoom_range=0.2,
                               width_shift_range=0.5,
                               height_shift_range=0.5
                               )   

validation_gen = ImageDataGenerator(horizontal_flip=True,
                               rotation_range=45,
                               zoom_range=0.2,
                               width_shift_range=0.5,
                               height_shift_range=0.5
                               )   

# Build the model
# ===========================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

batch_size = 16
epochs = 50

base = Xception(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base.output
x = GlobalAveragePooling2D()(x)

head = Dense(5, activation='softmax')(x)
                                      
model = Model(inputs=base.input, outputs=head)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model

history = model.fit(
    train_gen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data = validation_gen.flow(X_val, y_val, batch_size=batch_size),
    steps_per_epoch=X_train.shape[0] // batch_size,
)


# Display the results

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)


confmx = confusion_matrix(y_test_classes, y_pred_classes)
f , ax = plt.subplots(figsize=(8, 8)) 

sns.heatmap(confmx, annot=True, fmt='.1f' , ax=ax) 
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

model.save("/mnt/d/temp/models/ships_xception_model.keras")





