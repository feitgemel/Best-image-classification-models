import lightly_train

lightly_train.train(
    out="/mnt/d/temp/models/lightly-train/Object-Classification/out/9-dogs-Breed",            # Output directory
    data="/mnt/d/Data-Sets-Image-Classification/9 dogs Breeds",                 # Directory with images
    model="torchvision/resnet50",       # Model to train
    epochs = 100,
    batch_size = 32,
)