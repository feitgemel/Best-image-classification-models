import numpy as np
import cv2 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# load the saved model 
model_path = "/mnt/d/temp/models/ships_xception_model.keras"
model = load_model(model_path)

# load the image
image_path = "Best-image-classification-models/Xception-Ships-classifications/Cargo_Ship_Puerto_Cortes.jpg"
image = cv2.imread(image_path)

# Resize the image to the input size of the model
image_resized = cv2.resize(image, (128, 128)) 
image_array = img_to_array(image_resized)
image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

# Run prediction
predictions = model.predict(image_array)
predicted_class_idx = np.argmax(predictions, axis=1)[0]

# Map the predicted index to the corresponding ship type
categories = {0: 'Cargo' , 1: 'Military', 2: 'Carrier', 3: 'Cruise', 4: 'Tankers'}
predicted_class = categories[predicted_class_idx]


# Display the result
font = cv2.FONT_HERSHEY_SIMPLEX 
font_scale = 2 
font_color = (255,255,255) # white color
Thickness = 4 
line_type = cv2.LINE_AA 

# Get the text size for positioning 
text_size = cv2.getTextSize(predicted_class, font, font_scale, Thickness)[0]
text_x = 10 # constant x position
text_y = 100 # Position the text at the top-left corner

# Draw a cectangle behind the text for better visibility 
cv2.rectangle(image, (text_x - 10 , text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0,0,0), -1)

 # Put the text on the image
cv2.putText(image, predicted_class, (text_x, text_y), font, font_scale, font_color, Thickness, line_type)

# Display the image with the prediction
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()