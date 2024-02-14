import os 
import sys 
import cv2 
import csv 
import random 
import numpy as np
import torch
import ultralytics

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def split_data(dir_in, split_ratio):
    test_split, validation_split, train_split = split_ratio
    files = [f for f in os.listdir(dir_in) if f.endswith('.txt')]
    random.shuffle(files)

    for file in files:
        random_val = random.random()

        if random_val > 1 - test_split:
            split = 'test'
        elif random_val > 1 - test_split - validation_split:
            split = 'validation'
        else:
            split = 'train'

        image = file.replace('.txt', '.jpg')
        temp = dir_in.rfind('/')
        out_dir = os.path.join(dir_in[:temp], split)

        # move image and label to new directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        os.rename(os.path.join(dir_in, file), os.path.join(out_dir, "labels", file))
        os.rename(os.path.join(dir_in, image), os.path.join(out_dir, "images", image))
    print('Data split complete')
    
def train_model(data_path):
    model = ultralytics.YOLO('yolov8x.yaml').load("yolov8x.pt")
    print("Model loaded")
    model = model.to(device)
    results = model.train(
        data=data_path,
        project="/dcs/large/u2102661/CS310/models/tackle_location",
        epochs=400, 
        imgsz=640, 
        single_cls=True, 
        batch=8, 
        patience=100, 
        optimizer="Adam",
        close_mosaic=20, 
        lr0=0.001,
        freeze=10,
        dropout=0.3
    )
    return results

def main():
    # dir = sys.argv[1]
    # split_data(dir, (0.1, 0.1, 0.8))
    train_model("/dcs/large/u2102661/CS310/datasets/tackle_location/data.yaml")


if __name__ == '__main__':
    main()