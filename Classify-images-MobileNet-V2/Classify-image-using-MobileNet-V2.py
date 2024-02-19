# pip install tensorflow==2.10
# pip install numpy
# pip install opencv-python


# Python version : 3.9.16

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

model = MobileNetV2(weights = 'imagenet')
print(model.summary())
# classify 1000 classes -> imagenet

import numpy as np
import cv2

img = cv2.imread("Best-image-classification-models/Classify-images-MobileNet-V2/Dori.jpg")
print("Original shape :")
print(img.shape)

img = cv2.resize(img, (224,224))
print("resized shape : ")
print(img.shape)

# create a numpy array for the model 
data = np.empty((1, 224 , 224 , 3))

# store our image inside the "batch" of images

data[0] = img
print("Data shape for the model :")
print(data.shape)

# normelize the data between 0 and 1
data = preprocess_input(data)

# classify :

predictions = model.predict(data)
print (predictions)

# get the highest value :
highIndexValue = np.argmax(predictions , axis=1)
print(highIndexValue)

print("The predicted score value is :")
print(predictions[0][155])  # 89% 

# how to get the top 5 predicitions :
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

for name , desc , score in decode_predictions(predictions , top=5)[0] :
    print(desc , score)
