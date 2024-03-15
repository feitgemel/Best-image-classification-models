# Dataset : https://www.kaggle.com/datasets/yousefidris/ogedolympic-games-event-dateset

import numpy as np 
import cv2 
import os 

input_shape = (224,224)
path = "E:/Data-sets/olympics" 

categories = os.listdir(path)
categories.sort()

print(categories)
print(len(categories))
print ("==================================================")

def prepareData(path):
    Images = []
    Lables = []

    for category in categories:
        fullPath = os.path.join(path,category)
        #print(fullPath)

        file_names = os.listdir(fullPath)

        for file in file_names:
            file = os.path.join(fullPath, file)
            #print(file)
            img = cv2.imread(file)

            if img is not None:

                resized = cv2.resize(img , input_shape, interpolation = cv2.INTER_AREA)
                Images.append(resized)
                Lables.append(category)


    Images = np.array(Images)           
    #print(Images.shape)

    Lables = np.array(Lables)
    #print(Lables.shape)

    return Images , Lables


allImages , allLables = prepareData(path)
print(allImages.shape)
print(allLables.shape)

# show two images :

img_A , img_B = allImages[0],   allImages[17]
label_A , label_B = allLables[0], allLables[17]

print(label_A, label_B)

cv2.imshow("img1", img_A)
cv2.imshow("img2" , img_B)
cv2.waitKey(0)

print("Save the data .......")
np.save("e:/temp/olymp-images-224.npy", allImages)
np.save("e:/temp/olymp-labels-224.npy", allLables)
print("Finish save the data .......")


