import keras 
import keras_hub
import numpy as np
import matplotlib.pyplot as plt
import cv2 


# Load the model and use it to predict the image

# List of the models available in keras_hub : https://keras.io/keras_hub/presets/

model_name = "resnet_50_imagenet"
model_name2 = "resnet_vd_200_imagenet"

classifer = keras_hub.models.ImageClassifier.from_preset(model_name2, activation="softmax")

img_path = "Best-image-classification-models/Keras-Hub-Image-Classification/test_img.jpg"

image = keras.utils.load_img(img_path) 

preds = classifer.predict(np.array([image]))

print(keras_hub.utils.decode_imagenet_predictions(preds))

decoded_preds = keras_hub.utils.decode_imagenet_predictions(preds)

# Extract the top class name 
# The result is a list of tuples [(class_id , label , score), ......]
class_name = str(decoded_preds[0][0][0]) # Access the 'label' from the first prediction

# Load the image for display with OpenCV
image_cv = cv2.imread(img_path)
# Convert BGR to RGB
image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# Use Matplotlib to display the image with the predicted class name
plt.figure(figsize=(8, 8)) 
plt.imshow(image_cv)
plt.axis('off')
plt.title(f"Predicted: {class_name}", fontsize=20 , color='red')
plt.show()


