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
    # model = ultralytics.YOLO('yolov9e.yaml').load("yolov9e.pt")
    model = ultralytics.YOLO('yolov8x.yaml').load("yolov8x.pt")
    print("Model loaded")
    model = model.to(device)
    # ARGS = {
    #     "degrees": 15.0, # (float) image rotation (+/- deg)
    #     "translate": 0.1, # (float) image translation (+/- fraction)
    #     "scale": 0.5, # (float) image scale (+/- gain)
    #     "shear": 5.0, # (float) image shear (+/- deg)
    #     # perspective: 0.0 # (float) image perspective (+/- fraction), range 0-0.001
    #     "flipud": 0.0, # (float) image flip up-down (probability)
    #     "fliplr": 0.5, # (float) image flip left-right (probability)
    #     "mosaic": 1.0, # (float) image mosaic (probability)
    #     "mixup": 0.3, # (float) image mixup (probability)
    #     "copy_paste": 0.3, # (float) segment copy-paste (probability)
    #     # auto_augment: randaugment # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
    #     "erasing": 0.4, # (float) probability of random erasing during classification training (0-1)
    # }
    results = model.train(
        data=data_path,
        project="/dcs/large/u2102661/CS310/models/tackle_location",
        epochs=300, 
        imgsz=640, 
        single_cls=True, 
        pretrained=True,
        batch=32, 
        patience=50, 
        optimizer="Adam",
        close_mosaic=20, 
        lr0=0.001,
        # freeze=12,
        dropout=0.3, 
        # Data agumentation
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.3,
        erasing=0.4,
    )
    return results

def evaluate_model(input_path, output_path, model):
    # Make the videos directory if it doesn't exist
    os.makedirs(os.path.join(output_path, "videos")) if not os.path.exists(os.path.join(output_path, "videos")) else None

    #  Make a .txt file for an overview of the results
    with open(os.path.join(output_path, "results.txt"), "w") as f:
        for video in os.listdir(input_path):
            print("processing video: ", video)
            if not(video.endswith(".mp4")):
                continue
            video_path = os.path.join(input_path, video)
            cap = cv2.VideoCapture(video_path)
            print(cap.get(5))
            print(cap.get(3))
            print(cap.get(4))
            # Video writer
            video_writer = cv2.VideoWriter(os.path.join(output_path, "videos", video[:-4]) + "_tracked.mp4",
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        int(cap.get(5)), # This is the number of frames per second
                                        (int(cap.get(3)), int(cap.get(4))))
            assert cap.isOpened(), "Error reading video file"
            frame_number = 0
            # Make sure output directory exists
            os.makedirs(os.path.join(output_path, video[:-4])) if not os.path.exists(os.path.join(output_path, video[:-4])) else None
            # Read until video is completed
            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    break
                frame_name = os.path.join(output_path, video[:-4], video[:-4] + "_frame" + str(frame_number) + ".jpg")
                cv2.imwrite(frame_name, im0)
                named_image = cv2.imread(frame_name)
                results = model.predict(named_image, show=False, imgsz=640)
                # results = yolo.predict(named_image, show=False, imgsz=1280) 
                for r in results:
                    f.write("Tackle: " + video[:-4] + "  Frame: " + str(frame_number) + " Number of objects detected: " + str(len(r.boxes.xyxy)) + "\n")
                # f.write("Tackle: ", video[:-4], " Frame: ", frame_number, "Number of objects detected: ", len(results.xyxy[0]))
                annotated_frame = results[0].plot() 
                cv2.imwrite(frame_name[:-4] + "_tracked.jpg", annotated_frame)
                os.remove(frame_name)
                video_writer.write(annotated_frame)
                frame_number += 1
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()


def main():
    # dir = sys.argv[1]
    # split_data(dir, (0.1, 0.1, 0.8))
    train_model("/dcs/large/u2102661/CS310/datasets/tackle_location_new/data.yaml")
    # model = ultralytics.YOLO('yolov8x.yaml').load("/dcs/large/u2102661/CS310/models/tackle_location/train/weights/best.pt")
    # evaluate_model("/dcs/large/u2102661/CS310/datasets/anomaly_detection/tackle_videos", "/dcs/large/u2102661/CS310/model_evaluation/tackle_location", model)


if __name__ == '__main__':
    main()