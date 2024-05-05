from ultralytics import YOLO
import torch

def pose_detect(in_dir, out_dir, model):
    """
    This function uses the YOLO model to detect poses in a series of videos from a directory

    Input:
        in_dir: path to the directory containing the videos
        out_dir: path to the directory where the results will be saved
        model: the YOLO model to use for pose detection
    
    Output:
        results: the results of the pose detection (not used)
        A series of predicted videos saved in the out_dir
    """
    results = model(in_dir, show=False, save=True, project=out_dir, name='pose', classes=[0])
    return results

def main():
    """
    A simple test script to use the YOLO model to detect poses in a series of videos
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO('yolov8x-pose-p6.pt')
    pose_detect("data/output_set/trimmed", "data/output_set" , model)

if __name__ == "__main__":
    main()
