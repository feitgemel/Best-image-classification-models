import os 
import random
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications.resnet_v2 import preprocess_input


# Paths : 
test_path = "/mnt/d/Data-Sets-Image-Classification/Knee Osteoarthritis Dataset with Severity Grading/test"
model_path = '/mnt/d/temp/models/Knee Osteoarthritis_best_model.keras'

IMG_SIZE = (224, 224)
class_names = sorted(os.listdir(test_path))
print("Class names: ", class_names)

# load the saved model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully from disk.")

print("Model Summary: ")
print(model.summary())

# Task 1 : Predict a random image from the test set
def predict_random_image():

    # Select a random class folder
    random_class = random.choice(class_names)
    class_folder = os.path.join(test_path, str(random_class))

    # Randomly select an image from the class folder
    random_image_name = random.choice(os.listdir(class_folder))
    image_path = os.path.join(class_folder, random_image_name)

    print("Randomly selected image: ", image_path)

    # load the image using cv2
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    resized_image = cv2.resize(image, IMG_SIZE)

    # Add Debug for the preprocessed image
    print("Image shape before preprocessing: ", resized_image.shape)
    print("Image min/max before preprocessing: ", np.min(resized_image), np.max(resized_image))

    input_array = np.expand_dims(resized_image, axis=0) # Add batch dimension
    input_array = preprocess_input(input_array)

    print("Input array shape after preprocessing: ", input_array.shape) 
    print("Input array min/max after preprocessing: ", np.min(input_array), np.max(input_array))


    # Make prediction
    predictions = model.predict(input_array)
    predicted_class_index = np.argmax(predictions) 
    predicted_class = class_names[predicted_class_index]

    # Add debugging for predictions
    print("Raw model predictions: ", predictions)
    print("Predictions probabilities: ", predictions[0])
    print("Predicted class index: ", predicted_class_index)
    print("Predictions confidence scores: ", np.max(predictions) * 100 , "%")

    # Display the image with predicted class
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}\nTrue: {random_class}", fontsize=14)
    plt.axis('off')
    plt.show()

# Run the function to predict a random image
predict_random_image()
# ========================================================================================================


# Step 2 : Evaluate the model on the entire test set

def evaluate_model():
    true_labels = []
    predicted_labels = []

    for class_index , class_name in enumerate(class_names):
        class_folder = os.path.join(test_path, class_name)
        
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            
            #load the image using cv2
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #Preprocess the image
            resized_image = cv2.resize(image_rgb, IMG_SIZE)
            input_array = np.expand_dims(resized_image, axis=0) # Add batch dimension
            input_array = preprocess_input(input_array)

            # Make prediction
            predictions = model.predict(input_array)
            predicted_class_index = np.argmax(predictions)

            # Append to lists
            true_labels.append(class_index)
            predicted_labels.append(predicted_class_index)

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    print("Classification Report:\n", report)

# Run the evaluation
evaluate_model()
