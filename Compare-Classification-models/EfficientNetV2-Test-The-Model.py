import tensorflow as tf 
import cv2 
import os 

from keras.utils import load_img , img_to_array 
import numpy as np 
import time

start_time = time.time()

IMAGE_SIZE = 224

# get the classes 
imageFolder = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/train"
CLASSES = os.listdir(imageFolder)
num_classes = len(CLASSES)

best_model_file = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/MobileNet-V3-car.h5"
model = tf.keras.models.load_model(best_model_file)

def prepareImage(pathForImage) :
    image = load_img(pathForImage , target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result , axis=0)
    img_result = img_result / 255.
    return img_result


testImagePath = "Best-image-classification-models/Compare-Classification-models/chevrolet-cobalt-ss-supercharged.jpg"
img = cv2.imread(testImagePath)

ImgForModel = prepareImage(testImagePath)
resultArray = model.predict(ImgForModel , verbose=1)
answer = np.argmax(resultArray , axis=1)

print(answer)
index = answer[0]
desc = CLASSES[index]

print("The predicted class is : " + desc)


cv2.putText(img , desc , (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,0,0), 3 , cv2.LINE_AA)

end_time = time.time()
execution_time = end_time - start_time

print(f"The code took {execution_time:.6f} seconds to execute.")


cv2.imshow('img', img)
cv2.waitKey(0)