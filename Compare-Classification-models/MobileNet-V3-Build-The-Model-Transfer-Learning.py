# https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder

from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import time

start_time = time.time()


train_path = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/train"
validation_path = "C:/Data-sets/Stanford Car Dataset/car_data/car_data/test"

trainGenerator = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), batch_size=32)

validGenerator = ImageDataGenerator().flow_from_directory(validation_path, target_size=(224,224), batch_size=32)

# Build the model 
baseModel = MobileNetV3Large(weights= "imagenet", include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512 , activation='relu')(x)
x = Dense(256 , activation='relu')(x)
x = Dense(128 , activation='relu')(x)

predictLayer = Dense(196, activation='softmax')(x)

model = Model(inputs= baseModel.input , outputs=predictLayer)

print(model.summary())


# freeze the Mobilenet V3 pre-trained layers

for layer in model.layers[:-5] : # until the last layers we add
    layer.trainable = False


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