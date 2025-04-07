import lightly_train

lightly_train.train(
    out="/mnt/d/temp/models/lightly-train/out/my_experiment",            # Output directory
    data="/mnt/d/Data-sets/Playing-cards-YoloV8-Object-detection/train/images",                 # Directory with images
    model="torchvision/resnet50",       # Model to train
    epochs = 100,
    batch_size = 32,
)