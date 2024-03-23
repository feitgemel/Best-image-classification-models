import torch
from torchvision import transforms
from PIL import Image 
import cv2 

# load the Yolo-V5 model
weights_path = "C:/tutorials/yolov5/runs/train-cls/exp3/weights/best.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5','custom', path=weights_path, force_reload=True)
model.eval()

# define transformations 
transform = transforms.Compose([
    transforms.Resize((224,224)), # resize the test image to match the model input size
    transforms.ToTensor(),
])

# load the test image
#imagePath = "Best-image-classification-models/YoloV5/Images/Butterfly.jpg"
imagePath = "Best-image-classification-models/YoloV5/Images/El1.jpg"


img = Image.open(imagePath).convert('RGB')

# process the image and adjust it to our model
img = transform(img).unsqueeze(0).to(device)

# print the shape of the image tensor
print("Shape of the image tensor : ", img.shape)

# make the prediction
with torch.no_grad():
    outputs = model(img)

# get the predicted class index (Highest value)
predicted_class = torch.argmax(outputs, dim=1).item()

print(predicted_class)

# get the name of the class
import os 
categories = os.listdir("C:/Data-sets/Animals-10/raw-img/train")
print(categories)

text = "Predicted class : " + categories[predicted_class]
print(text)

# show the image with the predicted class
imgDisplay = cv2.imread(imagePath)
scale_precent = 35 
width = int(imgDisplay.shape[1] * scale_precent / 100 )
height = int(imgDisplay.shape[0] * scale_precent / 100 )

dim = (width, height)

# resize image
resized = cv2.resize(imgDisplay, dim , interpolation=cv2.INTER_AREA)
cv2.putText(resized, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0), 2)
cv2.imshow("test image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

