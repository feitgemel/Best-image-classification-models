import tensorflow as tf 
import cv2 
import os 
from keras.preprocessing import image 
from keras.utils import load_img, img_to_array
import numpy as np 

IMAGE_SIZE = 224

ImagesFolder = "C:/Data-sets/aerospace_images"
CLASSES = os.listdir(ImagesFolder)
num_classes = len(CLASSES)
print(CLASSES)

best_model_file ="e:/temp/air-vgg19.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())


def prepareImage(pathForImage) :
    image = load_img(pathForImage, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult , axis=0)
    imgResult = imgResult / 255. # normelize the data
    return imgResult

#testImagePth = "Best-image-classification-models/VGG19-Classify-Aerospace/Zeppellin.JPG"
testImagePth = "Best-image-classification-models/VGG19-Classify-Aerospace/Baloon.JPG"


img = cv2.imread(testImagePth)
imgForModel = prepareImage(testImagePth)

resultArray = model.predict(imgForModel , verbose=1)
answer = np.argmax(resultArray , axis=1)

print(answer)

index = answer[0]

className = CLASSES[index]

print("The predicted class is : " + className)

cv2.putText(img, className, (10,20) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
cv2.imshow("image", img)
cv2.waitKey(0)
