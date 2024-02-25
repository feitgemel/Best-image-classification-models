import os 
from tensorflow.keras.preprocessing import image 
from PIL import Image
import numpy as np 
import tensorflow as tf 
import cv2

# get the list of categories :
categories = os.listdir("E:/Data-sets/A Large Scale Fish Dataset/Fish_Dataset/dataset_for_model/train")
categories.sort()
print(categories)

# load the saved model :
modelSavedPath = "E:/Data-sets/A Large Scale Fish Dataset/Fish_Dataset/dataset_for_model/FishV3.h5"
model = tf.keras.models.load_model(modelSavedPath)

# predict the image 

def classify_image(imageFile):
    x= []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((320,320), Image.ANTIALIAS)

    x = image.img_to_array(img)
    x= np.expand_dims(x, axis=0)

    print(x.shape)
    pred = model.predict(x)
    print(pred)

    # get the higest prediction value 
    categoryValue = np.argmax(pred, axis=1)
    categoryValue = categoryValue[0]
    
    print(categoryValue)

    result = categories[categoryValue]

    return result


img_path = "Best-image-classification-models/Classify-Images-Transfer-Learning-MobileNet-V3/Sea-Bass-test.jpg"
resultText = classify_image(img_path)
print(resultText)

img = cv2.imread(img_path)
img = cv2.putText(img , resultText, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
