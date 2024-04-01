from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetV2S 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 

import time

start_time = time.time()
IMAGE_SIZE=224
num_classes = 196 

train_path = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/train"
validation_path = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/test"

trainGenerator = ImageDataGenerator().flow_from_directory(train_path, target_size=(IMAGE_SIZE,IMAGE_SIZE), batch_size=16)

validGenerator = ImageDataGenerator().flow_from_directory(validation_path, target_size=(IMAGE_SIZE,IMAGE_SIZE), batch_size=16)

# Build the model 
basemodel = EfficientNetV2S(weights='imagenet', input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False)
basemodel.trainable = True

model = tf.keras.Sequential([ 
    basemodel,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
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


path_for_saved_model = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/MobileNet-V3-car.h5"
model.save(path_for_saved_model)