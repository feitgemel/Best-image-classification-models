from ultralytics import YOLO

def main():

    # load the model 
    model = YOLO('e:/models/yolov8l-cls.pt')
    datasetPath = "c:/data-sets/Agricultural-pests"

    batch_size = 32
    project = "c:/data-sets/Agricultural-pests"
    experimnet = "My-model"

    results = model.train(data=datasetPath,
                          epochs=50,
                          project=project,
                          name= experimnet,
                          batch=batch_size,
                          device=0,
                          imgsz=224,
                          patience=5,
                          verbose=True,
                          val=True )

                          
if __name__ == "__main__":
    main()