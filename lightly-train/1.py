import lightly_train

if __name__ == "__main__":
  lightly_train.train(
      out="Best-Object-Detection-models/lightly-train/out/my_experiment",            # Output directory
      data="Best-Object-Detection-models/lightly-train/images",                 # Directory with images
      model="torchvision/resnet50",       # Model to train
      epochs = 100,
      batch_size = 8,
  )