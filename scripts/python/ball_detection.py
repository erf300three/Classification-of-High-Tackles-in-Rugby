from ultralytics import YOLO
import torch

# All of this code is for a failed attempt to train a YOLOv8 model on a dataset of images of rugby balls
# The model was trained on the dataset and its performance on images was acceptable. However it sturggled
#  to generalise to the video data and was not used in the final model. The code is left here for reference.

def predicting(in_path):
    """
    Function to predict the bounding boxes of the balls in the images in the input path

    Inputs:
        in_path: path to the folder containing the images to predict the bounding boxes of the balls in
    """
    # All of the models created have since been lost to make room for other useful models
    model = YOLO("yolov8x.pt")
    print("Model loaded")
    model.predict(in_path, save=True, show=False, persist=True)

def training(data_path):
    """
    A simple function to train the model on the data in the input path

    Inputs:
        data_path: path to the data to train the model on
    """
    model = YOLO('yolov8x.yaml').load("yolov8x.pt")
    print("Model loaded")
    results = model.train(data=data_path, epochs=400, imgsz=1280, single_cls=True, batch=8)
    return results


if __name__ == "__main__":
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    training("/dcs/large/u2102661/CS310/datasets/ball_detection_larger/data.yaml")
    predicting("/dcs/large/u2102661/CS310/datasets/ball_detection_larger/test/images")
