import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#load the saved model
best_model_file = "/mnt/d/temp/models/100-Sports-Model.keras"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

test_dir= '/mnt/d/Data-Sets-Image-Classification/100 Sports Image Classification/test'

#load the test dataset
val_datagen = ImageDataGenerator()
testing_dg = val_datagen.flow_from_directory(
    test_dir,
    class_mode='categorical',
    target_size=(224,224),
    batch_size=32,
    shuffle=False,
    seed=42)


predictions = model.evaluate(testing_dg)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get the true labels
true_labels = testing_dg.classes

# Use the model to predict the classes
predicted_classes = np.argmax(model.predict(testing_dg), axis=-1)

# Display the classification report
print("Classification Report:\n", classification_report(true_labels, predicted_classes, target_names=testing_dg.class_indices.keys()))


