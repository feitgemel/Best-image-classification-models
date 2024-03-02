import tensorflow as tf 
import cv2 
import os 

from keras.utils import load_img , img_to_array 
import numpy as np 

IMAGE_SIZE = 224

# get the classes 
imageFolder = 'e:/data-sets/olympics'
CLASSES = os.listdir(imageFolder)
num_classes = len(CLASSES)

best_model_file = 'e:/temp/olympics-EfficientNetV2.h5'
model = tf.keras.models.load_model(best_model_file)

def prepareImage(pathForImage) :
    image = load_img(pathForImage , target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result , axis=0)
    img_result = img_result / 255.
    return img_result


testImagePath = "Best-image-classification-models/EfficientNetv2-Build-Model-Custom-Dataset/waterpolo-test-image.png"
img = cv2.imread(testImagePath)

ImgForModel = prepareImage(testImagePath)
resultArray = model.predict(ImgForModel , verbose=1)
answer = np.argmax(resultArray , axis=1)

print(answer)
index = answer[0]
desc = CLASSES[index]

print("The predicted class is : " + desc)


scale_precent = 60
width = int(img.shape[1] * scale_precent / 100)
height = int(img.shape[0] * scale_precent / 100)
dim = (width, height)
resized = cv2.resize(img, dim , interpolation = cv2.INTER_AREA)

cv2.putText(resized , desc , (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,0,0), 3 , cv2.LINE_AA)

cv2.imwrite("e:/temp/waterpolo.png" , resized)

cv2.imshow('img', resized)
cv2.waitKey(0)