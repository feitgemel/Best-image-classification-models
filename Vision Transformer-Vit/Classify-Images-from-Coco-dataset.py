from transformers import ViTFeatureExtractor , ViTForImageClassification
from PIL import Image as img 
import cv2

originalImage = cv2.imread("Best-image-classification-models/Vision Transformer-Vit/Dori.jpg")
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

# Create Feature ectractor (form tasks like resize , normalize pixesl , and prepare the image for the model)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# get the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


# extract the features 

# return_tensor="pt" : this parameter specifies that the function should return a Pytorch tensor
inputs = feature_extractor(images = img , return_tensors="pt")

#inputs -> contain the preproceessed image data 
outputs = model(**inputs)

# the logits golds the raw scores fot each class before applying any activation function
logits = outputs.logits 

# predict
predicted_class_idx = logits.argmax(-1).item()

print(predicted_class_idx) # 155

className = model.config.id2label[predicted_class_idx]

print("Predicted class : " + className)

originalImage = cv2.putText(originalImage, className, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

cv2.imshow("img", originalImage)
cv2.waitKey(0)