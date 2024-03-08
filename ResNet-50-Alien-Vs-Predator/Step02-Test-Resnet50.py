import tensorflow as tf 
import cv2 
import os 
from keras.utils import load_img, img_to_array
import numpy as np 

IMAGE_SIZE = 224

trainMyImagesFolder = "E:/Data-sets/alien_vs_predator_thumbnails/data/train"
CLASSES = os.listdir(trainMyImagesFolder)
num_classes = len(CLASSES)

best_model_file = 'e:/temp/alien-predator-model.h5'
model = tf.keras.models.load_model(best_model_file)

def prepareImage(pathForImage) :
    image = load_img(pathForImage, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255. 
    return imgResult

#testImagePath = "Best-image-classification-models/ResNet-50-Alien-Vs-Predator/test-predator.jpg"
testImagePath = "Best-image-classification-models/ResNet-50-Alien-Vs-Predator/test-alien.jpg"


img = cv2.imread(testImagePath)

imgForModel = prepareImage(testImagePath)

resultArray = model.predict(imgForModel , verbose=1)
print(resultArray)

answer = np.argmax(resultArray , axis=1)
print(answer)

index = answer[0]
className = CLASSES[index]

print("The predicted class is : " + className)

cv2.putText(img , className , (220,20), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0) , 2, cv2.LINE_AA)

cv2.imshow("img", img)
cv2.waitKey(0)

