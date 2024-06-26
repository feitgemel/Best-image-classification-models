import tensorflow as tf 
import os 
import numpy as np 
import cv2 
import time

start_time = time.time()

best_model_file = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/Dense201-V2-car.h5"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

input_shape = (224,224)
batch_size = 32 
path = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/train"

categories = os.listdir(path)
categories.sort()

print(categories)
print(len(categories))

def prepareImage(img) :
    resized = cv2.resize(img, input_shape , interpolation = cv2.INTER_AREA)
    imgResult = np.expand_dims(resized , axis= 0)
    imgResult = imgResult / 255.
    return imgResult

# image form google 
#testImagePath = "Best-image-classification-models/DenseNet201-Olympic Games/Rugbi-Google.jpg"
testImagePath = "Best-image-classification-models/Compare-Classification-models/chevrolet-cobalt-ss-supercharged.jpg"


img = cv2.imread(testImagePath)

ImageForModel = prepareImage(img)

#run prediction
result = model.predict(ImageForModel , verbose=1)
print(result)

# get the index of the highest prediction value
answers = np.argmax(result , axis=1)
print(answers)

text = categories[answers[0]]
print("The predicted class is : " + text)

# show the image
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img , text , (20,20) , font , 1, (0,255,255), 2) # Yellow color
cv2.imshow("img", img)

end_time = time.time()
execution_time = end_time - start_time

print(f"The code took {execution_time:.6f} seconds to execute.")

cv2.waitKey(0)
cv2.destroyAllWindows()



