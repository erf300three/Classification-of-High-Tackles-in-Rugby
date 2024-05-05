import os
import random
import argparse
import cv2
import torch
import ultralytics

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def split_data(dir_in, split_ratio):
    """
    This function splits the data in a directory into training, validation and test sets by 
    moving the files into new directories

    Input:
        dir_in: path to the directory containing the data
        split_ratio: tuple containing the ratio of the data to be split into training, validation 
        and test sets

    Output:
        The data in the input directory will be split into training, validation and test sets and
        moved into new directories
    """

    test_split, validation_split, _ = split_ratio
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
    """
    This function trains a YOLOv8 or YOLOv9 model on the data in the data_path directory

    Input:
        data_path: path to the YAML file containing the data configuration
    
    Output:
        results: the results of the training the model (not used)
    """

    # Swap between YOLOv8 and v9 models to train
    # model = ultralytics.YOLO('yolov9e.yaml').load("yolov9e.pt")
    model = ultralytics.YOLO('yolov8x.yaml').load("yolov8x.pt")
    print("Model loaded")
    model = model.to(device)
    results = model.train(
        data=data_path,
        project="/dcs/large/u2102661/CS310/models/tackle_location",
        epochs=100,
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
    print("Training complete")
    return results

def evaluate_model(input_path, output_path, model):
    """
    This function evaluates the model on the videos in the input_path directory and saves the 
    results in the output_path directory

    Input:
        input_path: path to the directory containing the videos to be evaluated
        output_path: path to the directory where the results will be saved
        model: the model to be evaluated
    
    Output:
        The results of the evaluation will be saved in the output_path directory
    """

    # Make the videos directory if it doesn't exist
    if not os.path.exists(os.path.join(output_path, "videos")):
        os.makedirs(os.path.join(output_path, "videos"))

    #  Make a .txt file for an overview of the results
    with open(os.path.join(output_path, "results.txt"), "w", encoding="utf-8") as f:
        for video in os.listdir(input_path):
            print("processing video: ", video)
            if not video.endswith(".mp4"):
                continue
            video_path = os.path.join(input_path, video)
            cap = cv2.VideoCapture(video_path)
            video_settings = (
                cv2.VideoWriter_fourcc(*'mp4v'),
                int(cap.get(5)),
                (int(cap.get(3)), int(cap.get(4)))
            )
            # Video writer
            file_name = os.path.join(output_path, "videos", video[:-4] + "_tracked.mp4")
            video_writer = cv2.VideoWriter(file_name, *video_settings)
            assert cap.isOpened(), "Error reading video file"
            frame_number = 0
            # Make sure output directory exists
            if not os.path.exists(os.path.join(output_path, video[:-4])):
                os.makedirs(os.path.join(output_path, video[:-4]))
            # Read until video is completed
            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    print("Video frame is empty. Exiting...")
                    break
                frame_name = os.path.join(output_path, video[:-4], f"Frame_{str(frame_number)}.jpg")
                cv2.imwrite(frame_name, im0)
                named_image = cv2.imread(frame_name)
                # Make model prediction
                results = model.predict(named_image, show=False, imgsz=640)

                for r in results:
                    f.write(f"Tackle: {video[:-4]} Frame: {str(frame_number)} Number of objects: {str(len(r.boxes.xyxy))}\n")
                annotated_frame = results[0].plot()
                cv2.imwrite(frame_name[:-4] + "_tracked.jpg", annotated_frame)
                os.remove(frame_name)
                video_writer.write(annotated_frame)
                frame_number += 1
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()

def main():
    """
    This function is the main function that runs the data splitting, model training and model
    evaluation functions. It takes in command line arguments to determine the operation to perform
    and the paths to the data and model files

    Arguments:
        --dir: path to the directory containing the data
        --output: path to the directory where the results will be saved
        --op: operation to perform: split, train, evaluate
        --data: path to the YAML file containing the data configuration
        --model: path to the model file
    """

    parser = argparse.ArgumentParser(description="Train and evaluate a tackle localisation model")
    parser.add_argument("--dir", help="Path to the directory containing the data")
    parser.add_argument("--output", help="Path to the directory where the results will be saved")
    parser.add_argument("--op", help="Operation to perform: split, train, evaluate")
    parser.add_argument("--data", help="Path to the YAML file containing the data configuration")
    parser.add_argument("--model", help="Path to the model file")

    args = parser.parse_args()

    match args.op:
        case "split":
            split_data(args.dir, (0.1, 0.1, 0.8))
        case "train":
            train_model(args.data)
        case "evaluate":
            model = ultralytics.YOLO('yolov8x.yaml').load(args.model)
            evaluate_model(args.dir, args.output, model)
        case _:
            print("Invalid operation. Please select split, train or evaluate")
            exit(1)

if __name__ == '__main__':
    main()
