import os 
import random
import numpy as np
import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


test_path = '/mnt/d/Data-Sets-Image-Classification/30 Musical Instruments/test/'
model_path = "/mnt/d/models/Best-CNN-Model-30-Musical-Instruments.keras"

IMG_SIZE = (128,128)
class_names = sorted(os.listdir(test_path))
print(class_names)

# Load the trained model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Task 1 - Predict a random image from the test folder 

def predict_random_image():
    # Select a random class folder 
    random_class = random.choice(class_names)
    class_folder = os.path.join(test_path, random_class)

    # Select a random image from the class folder
    random_image = random.choice(os.listdir(class_folder))
    image_path = os.path.join(class_folder, random_image)

    # load the image using Opencv
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    image_resized = cv2.resize(image_rgb, IMG_SIZE)
    input_array = np.expand_dims(image_resized / 255.0, axis=0)  # Normalize and add batch dimension

    # Prediction
    predictions = model.predict(input_array)
    predicted_class_index = np.argmax(predictions) 
    predicted_class = class_names[predicted_class_index]

    # Dsiplay the image with predicted label
    plt.figure(figsize=(6,6))
    plt.imshow(image_rgb)
    plt.title(f"Predicted: {predicted_class}\nTrue Class: {random_class}", fontsize=14)
    plt.axis('off')
    plt.show()
                                                                                      
# Run the prediction function (random image)
predict_random_image()

# -----------------------------------------------------------------
# Task 2 : Predict all images in the test folder and display a confusion matrix

def evaluate_model():
    true_labels = []
    predicted_labels = []

    for class_index , class_name in enumerate(class_names):
        class_folder = os.path.join(test_path, class_name)
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)

            # Load and preprocess the image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, IMG_SIZE)
            input_array = np.expand_dims(image_resized / 255.0, axis=0)  # Normalize and add batch dimension

            # Prediction
            predictions = model.predict(input_array)
            predicted_class_index = np.argmax(predictions)
            
            # Append true and predicted labels
            true_labels.append(class_index)
            predicted_labels.append(predicted_class_index)


    # Generate confustion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Visualize the confusion matrix
    plt.figure(figsize=(12,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names) 
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    print("Classification Report:\n", report)
    print(f"Evaluated class: {class_name}")



# Run the evaluation function
evaluate_model()


