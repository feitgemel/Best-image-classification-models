# Dataset : https://www.kaggle.com/datasets/yousefidris/ogedolympic-games-event-dateset

import os 
import shutil 
import random 

# 90% of the dataset -> train folder 
# 10% of the dataset -> test folder 

# set the path to the main dataset folder :
main_folder = 'e:/data-sets/olympics'

# set path for train and test

train_folder = 'e:/data-sets/olympics-train'
test_folder = 'e:/data-sets/olympics-test'

os.makedirs(train_folder , exist_ok=True)
os.makedirs(test_folder , exist_ok=True)

# get the list of subfolders
subfolders = []

for f in os.scandir(main_folder):
    if f.is_dir():
        subfolders.append(f.path)

# split train / test
train_precentage = 90


for subfolder in subfolders:

    # extract folder name 
    subfolder_name = os.path.basename(subfolder)

    # create the train and test subfolder
    train_subfolder = os.path.join(train_folder , subfolder_name)
    test_subfolder = os.path.join(test_folder , subfolder_name)

    os.makedirs(train_subfolder , exist_ok=True)
    os.makedirs(test_subfolder , exist_ok=True)

    # list all files in the current subfolder

    files = [f.path for f in os.scandir(subfolder) if f.is_file()]

    # suffle the files
    random.shuffle(files)

    # calculate the number of train files
    num_train_files = int( len(files) * (train_precentage / 100) )

    # copy the files to the train folder
    for file in files[:num_train_files]:
        shutil.copy(file,os.path.join(train_subfolder , os.path.basename(file)))

    # copy the files to the test folder :
    for file in files[num_train_files:]:
        shutil.copy(file,os.path.join(test_subfolder , os.path.basename(file)))       


print("Finish copy the files to Train and Test subfolder")