import tensorflow as tf
import cv2
import os

from keras.preprocessing import image
from keras.utils import img_to_array, load_img
import numpy as np

IMAGE_SIZE = 224

best_model_file = "/mnt/d/temp/models/100-Sports-Model.keras"
model = tf.keras.models.load_model(best_model_file)

test_dir= '/mnt/d/Data-Sets-Image-Classification/100 Sports Image Classification/test'
CLASSES = os.listdir(test_dir)
print(CLASSES)
num_classes = len(CLASSES)
print(f"Number of classes: {num_classes}")

def prepareImage(pathForImage): 
    image = load_img(pathForImage, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    imgResult = img_to_array(image) 
    imgResult = np.expand_dims(imgResult, axis=0)
    #imgResult = imgResult / 255.0
    return imgResult


testImagePath = "/mnt/d/Data-Sets-Image-Classification/100 Sports Image Classification/test/bike polo/2.jpg"

img = cv2.imread(testImagePath)
imgForModel = prepareImage(testImagePath)


resultArray = model.predict(imgForModel, verbose=1)
answer = np.argmax(resultArray, axis = 1) 

# Get the class label from the index
index = answer[0]
print(f"Predicted class index: {index}")
predicted_class = CLASSES[index]
print(f"Predicted class label: {predicted_class}")

cv2.putText(img, predicted_class , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
cv2.imshow("Prediction", img)
cv2.waitKey(0)





