import cv2 
import numpy as np

# load image 
imagePath = "Best-image-classification-models/Media-Pipe-Object-Classification/Dori.jpg"
img = cv2.imread(imagePath)
cv2.imshow("Image", img)

# Step 1 - import the necessary libraries
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Step2 - Create an Image Classifier

base_options = python.BaseOptions(model_asset_path="D:/Temp/Models/efficientnet_lite0.tflite")
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4) 
classifier = vision.ImageClassifier.create_from_options(options)

image = mp.Image.create_from_file(imagePath)

classification_result = classifier.classify(image)

top_category = classification_result.classifications[0].categories[0]
category_name = top_category.category_name
score = round(top_category.score, 2)

print(f"Category Name: {category_name}")
print(f"Score: {score}")

result_text  = f"Category Name: {category_name} , Score: {score}"
imageClassify = np.copy(img)

TEXT_COLOR = (255,0,0) # red 
FONT_SIZE = 2
FONT_THICKNESS = 3

cv2.putText(imageClassify, result_text, (10,50), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
cv2.imwrite("Best-image-classification-models/Media-Pipe-Object-Classification/Dori-out.jpg", imageClassify)
cv2.imshow("Image Classify", imageClassify)

cv2.waitKey(0)
cv2.destroyAllWindows()

