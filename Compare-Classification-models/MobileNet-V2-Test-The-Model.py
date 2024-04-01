import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 , preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time


start_time = time.time()

# get the list of classes 
categories = os.listdir("C:/Data-sets/Stanford Car Dataset/car_data/car_data/train")
categories.sort()
print(categories)

#load the saved model
path_for_saved_model = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/MobileNet-V2-car.h5"
model = tf.keras.models.load_model(path_for_saved_model)

#print(model.summary())

def classify_image(imageFile):
    x = []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((224,224), Image.ANTIALIAS)

    x = image.img_to_array(img)
    x = np.expand_dims(x , axis=0)
    x = preprocess_input(x)
    print(x.shape)

    pred = model.predict(x)
    categoryValue = np.argmax(pred , axis=1)
    print(categoryValue)

    categoryValue = categoryValue[0]
    print(categoryValue)   

    result = categories[categoryValue]

    return result

imagePath = "Best-image-classification-models/Compare-Classification-models/chevrolet-cobalt-ss-supercharged.jpg"
resultText = classify_image(imagePath)
print(resultText)

img = cv2.imread(imagePath)
img = cv2.putText(img , resultText , (50,50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

cv2.imshow("img", img)
end_time = time.time()
execution_time = end_time - start_time

print(f"The code took {execution_time:.6f} seconds to execute.")

cv2.waitKey(0)
cv2.destroyAllWindows()


