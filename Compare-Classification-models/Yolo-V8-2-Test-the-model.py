from ultralytics import YOLO 
import numpy as np 
import cv2 
import time

start_time = time.time()

# load the model 
model = YOLO("C:/Data-sets/Stanford Car Dataset/car_data/Nano-224/weights/best.pt")
imgPth = "Best-image-classification-models/Compare-Classification-models/chevrolet-cobalt-ss-supercharged.jpg"

# predict 
results = model(imgPth)

#print(results)

names_dict = results[0].names
print("Categories : ")
print(names_dict)
print("*************************************************")

probs = results[0].probs.data.tolist()
print("All predictions :")
print(probs)
print("*************************************************")

print("The predicted class is : ")
text = "Predicted category :" + names_dict[np.argmax(probs)]

print(text)

#display the image

imgDisplay = cv2.imread(imgPth)
cv2.putText(imgDisplay, text , (10,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("img", imgDisplay)

end_time = time.time()
execution_time = end_time - start_time

print(f"The code took {execution_time:.6f} seconds to execute.")

cv2.waitKey(0)
cv2.destroyAllWindows()