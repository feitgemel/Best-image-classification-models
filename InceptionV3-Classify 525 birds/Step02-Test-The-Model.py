import tensorflow as tf 
import cv2 
import os 

from keras.utils import load_img , img_to_array
import numpy as np 

IMAGE_SIZE = 224

# GET CLASSES NAMES
pathToDataset = "E:/Data-sets/BIRDS 525 SPECIES- IMAGE CLASSIFICATION"
trainPath = pathToDataset + "/train"
testPath = pathToDataset + "/test"
validPath = pathToDataset + "/valid"

CLASSES = os.listdir(trainPath)
no_of_classes = len(CLASSES)
print("No. of Classes : " + str(no_of_classes)) 

best_model_file = "e:/temp/525-birds.h5"
model = tf.keras.models.load_model(best_model_file)

print(model.summary())

def prepareImage(pathForImage) :
    image = load_img(pathForImage, target_size=(IMAGE_SIZE,IMAGE_SIZE))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis = 0)
    imgResult = imgResult /  255.
    return imgResult

# get test image
#testImagePath = testPath + "/ALBATROSS/4.jpg"
testImagePath = testPath + "/ANHINGA/5.jpg"


img = cv2.imread(testImagePath)
imgForModel = prepareImage(testImagePath)
resultArray = model.predict(imgForModel, verbose=1)
answer = np.argmax(resultArray, axis = 1)

print(answer)
index = answer[0]

print("The predicted class is : " + CLASSES[index])

cv2.putText(img , CLASSES[index], (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,0,0), 1, cv2.LINE_AA)
cv2.imshow("img", img)
cv2.waitKey(0)



