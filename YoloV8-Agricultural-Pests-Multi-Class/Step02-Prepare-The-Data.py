# Dataset : https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset

# Download the dataset and extract it in a new folder

import os
import shutil
import random
import cv2
  


def split_images(input_folder, output_folder, split_ratio=0.9):
    # Create train and validate folders
    train_folder = os.path.join(output_folder, 'train')
    validate_folder = os.path.join(output_folder, 'val')
    num = 0

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validate_folder, exist_ok=True)

    # Get the list of subfolders in the input folder
    subfolders = [f.name for f in os.scandir(input_folder) if f.is_dir()]

    # Iterate over each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        train_subfolder_path = os.path.join(train_folder, subfolder)
        validate_subfolder_path = os.path.join(validate_folder, subfolder)

        os.makedirs(train_subfolder_path, exist_ok=True)
        os.makedirs(validate_subfolder_path, exist_ok=True)

        # Get the list of images in the subfolder
        images = [f.name for f in os.scandir(subfolder_path) if f.is_file()]

        # Calculate the number of images to be moved to the validation set
        num_images = len(images)
        num_validate = int(num_images * (1 - split_ratio))

        # Randomly select images for the validation set
        validate_images = random.sample(images, num_validate)

        # Move images to the respective folders
        for image in images:
            source_path = os.path.join(subfolder_path, image)
            #print(source_path)

            # try to load the image check validity and copy only good images
            img = cv2.imread(source_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # a small change before save
            
            # generate a new file name for each image using numerator
            name = str(num) + ".png"

            # "copy" the image using cv2.imwrite
            if img is not None:
                if image in validate_images:
                    #destination_path = os.path.join(validate_subfolder_path, image)
                    destination_path = os.path.join(validate_subfolder_path, name) 
                    cv2.imwrite(destination_path,img)
                else:
                    #destination_path = os.path.join(train_subfolder_path, image)
                    destination_path = os.path.join(train_subfolder_path, name) 
                    cv2.imwrite(destination_path,img)

            else :
                print("Invalid image or file not found.")

            #shutil.copy2(source_path, destination_path)
            print(destination_path)
            num = num + 1


# run the code:
        
input_folder = "E:/Data-sets/Agricultural-Pests"
output_folder = "C:/Data-sets/Agricultural-Pests"

split_images(input_folder, output_folder)



