from ultralytics import YOLO
import numpy as np 
import cv2 


model = YOLO("C:/Data-sets/Agricultural-Pests/My-model/weights/best.pt")

imgPath = 'Best-image-classification-models/YoloV8-Agricultural-Pests-Multi-Class/Garden-Snail.jpg'
results = model(imgPath)

# get classes names
names = results[0].names 

# get preictions results
probs = results[0].probs.data.tolist()

print("Categories : ")
print(names)

print("All predicitions :")
print(probs)

best_prediction = np.argmax(probs)
best_prediction_names = names[best_prediction]

print ("Best prediction :")
print(best_prediction)
print(best_prediction_names)

imgDisplay = cv2.imread(imgPath)

# display the text with the image
cv2.putText(imgDisplay, best_prediction_names , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0) , 2)
cv2.imshow("img", imgDisplay)
cv2.waitKey(0)
cv2.destroyAllWindows()


