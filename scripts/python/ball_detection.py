import os 
from ultralytics import YOLO
import torch

def predicting(in_path):
    model = YOLO("/dcs/21/u2102661/Documents/3rdYear/CS310/Classification-of-High-Tackles-in-Rugby/runs/detect/train/weights/best.pt")
    # model = YOLO("yolov8x.pt")
    print("Model loaded")
    model.predict(in_path, save=True, show=False, persist=True)

def training(data_path):
    model = YOLO('yolov8x.yaml').load("yolov8x.pt")
    print("Model loaded")
    results = model.train(data=data_path, epochs=400, imgsz=1280, single_cls=True, batch=8)
    return results


if __name__ == "__main__":
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    results = training("/dcs/large/u2102661/CS310/datasets/ball_detection_larger/data.yaml")
    # predicting("/dcs/large/u2102661/CS310/datasets/ball_detection_larger/test/images")