#Link for documentation : https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

from tensorflow.keras.applications import EfficientNetB0
import cv2
from tensorflow.keras.preprocessing import image 
import numpy as np 

model = EfficientNetB0(weights='imagenet') # classify 1000 classes

originalImg = cv2.imread("Best-image-classification-models/EfficientNet-image-Classification/Dori.jpg")
print(originalImg.shape) # (1600, 1200, 3)

img = cv2.resize(originalImg, (224,224)) # Model B0 requires : 224X224 input shape
print(img.shape)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
print(pred)
print(pred.shape)

# get the description of the best high score prediction :

from tensorflow.keras.applications.imagenet_utils import decode_predictions

for name , desc , score in decode_predictions(pred , top=1)[0] :
    print(desc , score)

#Result : Shih-Tzu 0.74700737
    
img = cv2.putText(originalImg , desc , (50,50) , cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0), 2)
cv2.imshow("img", originalImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

