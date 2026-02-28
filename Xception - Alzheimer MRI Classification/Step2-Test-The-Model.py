import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

import matplotlib.pyplot as plt
import os 
import random

SIZE = 299 # Xception model input size 299X299 pixels

def predict_random_image(model_path , dataset_path):

    # Defince class folders based on actual dataset structure
    class_folders = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

    # map folder names to display labels if needed
    class_display_labels = {
        'MildDemented': 'Mild Demented',
        'ModerateDemented': 'Moderate Demented', 
        'NonDemented': 'Non Demented',
        'VeryMildDemented': 'Very MildDemented'
    }

    try :
        # load the saved model
        print (f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)

        # Select random class folder
        random_class = random.choice(class_folders)
        class_path = os.path.join(dataset_path, random_class)

        # Get tje random image from the selected class folder
        image_files = os.listdir(class_path)
        random_image = random.choice(image_files)
        image_path = os.path.join(class_path, random_image)

        print(f"Selected random image: {image_path}")
        print(f"True class: {class_display_labels[random_class]}")

        # load and preprocess the image
        img = image.load_img(image_path, target_size=(SIZE, SIZE))
        img_array = image.img_to_array(img) # convert to array
        img_array = np.expand_dims(img_array, axis=0) # add batch dimension
        processed_img = preprocess_input(img_array) # preprocess for Xception

        # make prediction
        prediction = model.predict(processed_img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Get predicted label (using the same order as in your training data)
        model_class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        predicted_class = model_class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100

        # Display the image with both true and predicted labels
        plt.figure(figsize=(10,10))
        plt.imshow((img_array[0] + 1) / 2) # un-preprocess for display

        # Set color based on correctness
        color = "green" if class_display_labels[random_class] == predicted_class else "red"

        # Add both true and predicted labels to the title
        plt.title(f"True: {class_display_labels[random_class]}\nPredicted: {predicted_class} (Confidence: {confidence:.2f}%)", 
                  color=color, fontsize=14)
        plt.axis("off")
        plt.show()


        # Print detailed prediction results
        print("\nPrediction Results:")
        print(f"True Class: {class_display_labels[random_class]}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Pedicted Correct: {'Yes' if class_display_labels[random_class] == predicted_class else 'No'}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nDetailed Class Probabilities:")
        for i , label in enumerate(model_class_labels):
            print(f"{label}: {prediction[0][i] * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the prediction function with the path to your saved model and dataset
if __name__ == "__main__":

    # Update these paths to your actual model and dataset locations
    model_path = "/mnt/d/temp/models/Alzheimer_Model.keras"
    dataset_path = r"/mnt/d/Data-Sets-Image-Classification/Augmented Alzheimer MRI Dataset/AugmentedAlzheimerDataset"

    # verify if the patsh exist before running the prediction
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
    elif not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
    else:
        # Run the random image prediction
        predict_random_image(model_path, dataset_path)


        































   












































