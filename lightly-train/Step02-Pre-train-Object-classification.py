import lightly_train

lightly_train.train(
    data="/mnt/d/Data-Sets-Image-Classification/9 dogs Breeds",
    out="/mnt/d/temp/models/lightly-train/Object-Classification/out/9-dogs-Breed",
    model="torchvision/resnet50",
    epochs=100,
    batch_size=32,
)

