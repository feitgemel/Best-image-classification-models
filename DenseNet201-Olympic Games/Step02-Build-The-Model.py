import numpy as np 
import tensorflow as tf 


#load the saved data 
allImages = np.load("e:/temp/olymp-images-224.npy")
allLabels = np.load("e:/temp/olymp-labels-224.npy")

print(allImages.shape)
print(allLabels.shape)

#show a sample image
import cv2 
img = allImages[0]
label = allLabels[0]

print(label)
#cv2.imshow("img", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#normalize the images between 0 to 1
allImagesForModel = allImages / 255.0


# Split train and test data
from sklearn.model_selection import train_test_split
print("Start train and test split data :")
X_train, X_test , y_train , y_test = train_test_split(allImagesForModel, allLabels, test_size=0.3, random_state=42)

print("X_train , X_test , y_train , y_test   ----->>> shapes :")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# convert the lables to numbers and also convert to hot encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# lable encode the training lables 
y_train_encoded = label_encoder.fit_transform(y_train)

# lable encode the test lables
y_test_encoded = label_encoder.fit_transform(y_test)


# convert labels to one-hot encoded format
y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded)

#print(y_test_one_hot)

# Build the model :

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers, models

batch_size = 16
img_height , img_width = 224,224
num_classes = 10

base_model = DenseNet201(include_top=False , weights='imagenet', input_shape=(img_height,img_width,3))

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes , activation='softmax'))

epochs=1000
lr = 1e-4
opt = tf.keras.optimizers.Adam(lr)

print(model.summary())

# ----------------------------------------------------------

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

best_model_file = "e:/temp/olymp-10.h5"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True , monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_accuracy", patience=5 , factor=0.1 , verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=20 , verbose=1) ]


# compile the model
model.compile(optimizer=opt , loss="categorical_crossentropy", metrics=['accuracy'])

#train the model 
hist = model.fit(
    X_train, y_train_one_hot, 
    steps_per_epoch = (len(X_train) / batch_size),
    validation_steps = (len(X_test) / batch_size),
    epochs=epochs,
    batch_size = batch_size,
    shuffle = True,
    validation_data = (X_test , y_test_one_hot),
    callbacks = callbacks
)



# print the highest validation accuracy 
highest_val_accuracy = max(hist.history["val_accuracy"])
print(f"Highest Validation Accuracy : {highest_val_accuracy}")











