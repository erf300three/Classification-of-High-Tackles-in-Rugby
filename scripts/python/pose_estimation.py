import os
from ultralytics import YOLO
import torch

def pose_detect(dir, out_dir, model):
        results = model(dir, show=False, save=True, project=out_dir, name='pose', classes=[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO('yolov8x-pose-p6.pt')

pose_detect("data/output_set/trimmed", "data/output_set" , model) 