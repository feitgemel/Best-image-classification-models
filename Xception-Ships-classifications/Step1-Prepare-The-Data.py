# Download dataset : https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets/data

import cv2 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import numpy as np

# Load the daat from the csv file
path_to_train_csv = '/mnt/d/Data-Sets-Image-Classification/Ships-dataset/train/train.csv'

data_csv = pd.read_csv(path_to_train_csv)
print("data shape: ", data_csv.shape)

categories = {0: 'Cargo' , 1: 'Military', 2: 'Carrier', 3: 'Cruise', 4: 'Tankers'}

data_csv['category'] = data_csv['category'] - 1 
data_csv['label'] = data_csv['category'].map(categories)
data_csv['label'] = pd.Categorical(data_csv['label'])

print(data_csv.head())

# ======================================================================

# How the data in a chart 
sns.countplot(data_csv['label'])
plt.title("Ship Category distribution")
plt.xlabel("category")
plt.ylabel("count")
plt.show()

# ======================================================================

path_Train_images = "/mnt/d/Data-Sets-Image-Classification/Ships-dataset/train/images"

img_list = list(data_csv['image'])

data_img = [] 

for img in img_list:
    print("img: ", img)
    each_path = os.path.join(path_Train_images, img)
    each_img = cv2.imread(each_path)
    each_img = cv2.cvtColor(each_img, cv2.COLOR_BGR2RGB)
    each_img_resized = cv2.resize(each_img, (128, 128))
    data_img.append(each_img_resized)


X = np.array(data_img)
print("X shape: ", X.shape)

# Save array to disk
np.save("/mnt/d/temp/data/ships_data.npy", X)

