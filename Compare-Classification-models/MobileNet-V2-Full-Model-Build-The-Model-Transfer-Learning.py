from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 , preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import time

start_time = time.time()


train_path = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/train"
validation_path = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/test"

trainGenerator = ImageDataGenerator(preprocessing_function = preprocess_input).flow_from_directory(train_path, target_size=(224,224), batch_size=32)
validGenerator = ImageDataGenerator(preprocessing_function = preprocess_input).flow_from_directory(validation_path, target_size=(224,224), batch_size=32)

# Build the model 
baseModel = MobileNetV2(weights='imagenet', include_top = False) # crop the last layer

baseModel.trainable = True

model = tf.keras.Sequential([ 
    baseModel,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(196, activation='softmax')
])


print(model.summary())



# compile
    
epochs = 50
optimizer = Adam(learning_rate = 0.0001)

model.compile(loss="categorical_crossentropy", optimizer=optimizer , metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# train
model.fit(trainGenerator , validation_data=validGenerator , epochs=epochs , callbacks=[early_stopping])

end_time = time.time()
execution_time_seconds  = end_time - start_time
execution_time_minutes = execution_time_seconds / 60

print(f"The code took {execution_time_minutes:.6f} minutes to execute.")


path_for_saved_model = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/MobileNet-V2-car.h5"
model.save(path_for_saved_model)