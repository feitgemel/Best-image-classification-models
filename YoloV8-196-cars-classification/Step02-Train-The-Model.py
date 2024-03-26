from ultralytics import YOLO

def main():

    model = YOLO('e:/models/yolov8l-cls.pt') # load the Yolo Large pre-trained model

    datasetPath = "C:/Data-sets/Stanford Car Dataset/car_data/car_data"

    batch_size = 16
    project = "C:/Data-sets/Stanford Car Dataset/car_data"
    experiment = "My-model"

    results = model.train(data=datasetPath,
                          epochs=30,
                          project=project,
                          name=experiment,
                          batch = batch_size,
                          device = 0,
                          imgsz=640,
                          patience=5,
                          verbose=True,
                          val=True)
    


if __name__ == "__main__" :
    main()