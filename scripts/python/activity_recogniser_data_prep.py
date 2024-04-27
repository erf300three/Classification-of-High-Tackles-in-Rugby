"""
This file contains the code required to generate the datasets for the action recognition model. The dataset is created
by taking the original video clips and creating new clips that are 5 or 16 frames long. The clips are labelled using 
the tackle bounds provided in the tackles.csv file. The clips are labelled as a tackle if at least 50% of the frames for 
16 frame clips and 60% of the frames for 5 frame clips contain a tackle. 
"""
import os
import csv
import random
import argparse
import cv2

def transform_csv_to_object(in_path):
    """
    This function reads the tackles.csv file and transforms it into a dictionary object containing the start and 
    end frames of the tackles for each video. This object is required to create the video clips and label them 
    accordingly.

    Input: 
        in_path: path to the directory containing the tackles.csv file

    Output:
        data: dictionary object containing the start and end frames of the tackles for each video
    """
    output_file = os.path.join(in_path, "tackles.csv")
    data = {}
    with open(output_file, 'r', newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["video_name"] not in data:
                data[row["video_name"]] = []
            data[row["video_name"]].append((int(row["start_frame"]), int(row["end_frame"])))
    return data

def create_tackle_videos(in_path, out_dir, data):
    """
    This function creates video clips of the tackles in the dataset. The clips are created by taking the frames
    from the tackle bounds and then creating new clips that start and end at the tackle bounds. 

    Input:
        in_path: path to the directory containing the original video clips
        out_dir: path to the directory where the tackle clips will be saved
        data: dictionary object containing the start and end frames of the tackles for each video
    
    Output:
        The clips containing only the tackle frames are saved in the out_dir/tackles directory
    """
    for video in os.listdir(in_path):
        print(f"===================={video}====================")
        if not video.endswith(".mp4"):
            continue
        video_path = os.path.join(in_path, video)
        cap = cv2.VideoCapture(video_path)
        video_settings = (
            cv2.VideoWriter_fourcc(*'mp4v'),
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )
        list_of_tackles = data[video]
        if not os.path.exists(os.path.join(out_dir, "tackles")):
            os.makedirs(os.path.join(out_dir, "tackles"))
        for idx, (tackle_start, tackle_end) in enumerate(list_of_tackles):
            outpath = os.path.join(out_dir, "tackles", video[:-4] + "_" + str(idx) + ".mp4")
            print(outpath)
            writer = cv2.VideoWriter(outpath, *video_settings)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, tackle_start - (cap.CAP_PROP_FPS * 3)))
            tackle_end = min(tackle_end + (cap.get(cv2.CAP_PROP_FPS) * 2), cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret = True
            while cap.isOpened() and ret and writer.isOpened():
                ret, frame = cap.read()
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if frame_number <= tackle_end:
                    writer.write(frame)
                if frame_number > tackle_end:
                    break
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

# We label a clip as a tackle if at least 50% of the frames in the clip contain a tackle
def create_videos(in_path, out_dir, data, clip_length=16, frame_overlap=8):
    """
    This function creates video clips of the original video. The length of these clips is determined by the clip_length
    parameter. The clips are labelled as a tackle if at least the number of frames specified by the frame_overlap 
    parameter are within the defined tackle bounds.
    This function also deals with resizing the video clips to 224x224 to match the input size of the Kinetics 400 
    dataset.

    Input:
        in_path: path to the directory containing the original video clips
        out_dir: path to the directory where the video clips will be saved
        data: dictionary object containing the start and end frames of the tackles for each video
        clip_length: length of the video clips to be created
        frame_overlap: number of frames required to be within the tackle bounds to label the clip as a tackle
    
    Output:
        The shorter video clips are saved in the out_dir/train, out_dir/validation and out_dir/test directories
    """

    for video in os.listdir(in_path):
        print(video)
        if not video.endswith(".mp4"):
            continue
        video_path = os.path.join(in_path, video)
        cap = cv2.VideoCapture(video_path)

        total_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(total_number_of_frames)
        list_of_all_frames = list(range(total_number_of_frames))
        list_of_clip_bounds = []
        # Need to create a list of clip bounds for the video
        list_of_clip_bounds = list(zip(*[iter(list_of_all_frames)]*clip_length))
        list_of_clip_bounds = [(x[0], x[-1]) for x in list_of_clip_bounds]
        print(list_of_clip_bounds)
        list_of_tackles = data[video]
        print(list_of_tackles)
        for idx,  (clip_start, clip_end) in enumerate(list_of_clip_bounds):
            is_tackle = 0
            for (tackle_start, tackle_end) in list_of_tackles:
                overlap = range(max(clip_start, tackle_start), min(clip_end, tackle_end))
                # We require the overlap to be greater than or equal to the frame_overlap to label the clip as a tackle
                if len(overlap) >= frame_overlap:
                    print((clip_start, clip_end))
                    print(len(overlap))
                    is_tackle = 1
            # Pick if clip is being plaed in the training, validation or test set with a 80/10/10 split
            random_float = random.random()
            location = "test" if random_float > 0.9 else "validation" if random_float > 0.8 else "train"
            csv_path = os.path.join(out_dir, location, "labels.csv")
            outpath = os.path.join(out_dir, location, video[:-4] + "_" + str(idx) + ".mp4")
            # print(outpath)
            with open(csv_path, 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([video[:-4] + "_" + str(idx) + ".mp4", clip_start, clip_end, is_tackle])
            # Write the video clip to a new file
            # Resizing the video to 224x224 to match the input size of kinematics 400 dataset
            writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            ret = True
            while cap.isOpened() and ret and writer.isOpened():
                ret, frame = cap.read()
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if frame_number <= clip_end:
                    frame = cv2.resize(frame, (224, 224))
                    writer.write(frame)
                if frame_number > clip_end:
                    break
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


def get_number_of_tackle_clips(in_path):
    """
    Returns the number of tackle clips and non-tackle clips in the dataset
    
    Input:
        in_path: path to the directory containing the dataset
    
    Output:
        number_of_tackle_clips: number of tackle clips in the dataset
        number_of_non_tackle_clips: number of non-tackle clips in the dataset
        total_number_of_clips: total number of clips in the dataset
    """
    number_of_tackle_clips = 0
    total_number_of_clips = 0
    # Select which directories to look at
    for dr in ["validation", "train", "test"]:
        with open(os.path.join(in_path, dr, "labels.csv"), 'r', newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_number_of_clips += 1
                if row["is_tackle"] == "1":
                    number_of_tackle_clips += 1

    return number_of_tackle_clips, total_number_of_clips - number_of_tackle_clips, total_number_of_clips

# Details about the number of clips in each dataset can be found in the action_recogniser_model.py file

def main():
    """
    Main function to parse the arguments and call the required functions to create the dataset for the action recogniser
    model. Additional functionality is provided to count the number of tackle clips in the dataset.

    Arguments:
        --in_path: path to the directory containing the original video clips
        --out_dir: path to the directory where the video clips will be saved
        --op: operation to perform: create_videos or create_tackle_videos
        --clip_length: length of the video clips to be created
        --frame_overlap: number of frames required to overalp the tackle bounds
    """
    parser = argparse.ArgumentParser(description="Create the dataset for the action recogniser model")
    parser.add_argument("--in_path", type=str, help="Path to the directory containing the original video clips")
    # default="/dcs/large/u2102661/CS310/datasets/activity_recogniser/original_clips"
    parser.add_argument("--out_dir", type=str, help="Path to the directory where the video clips will be saved")
    # default="/dcs/large/u2102661/CS310/datasets/activity_recogniser_5_frames"
    parser.add_argument("--op", type=str, help="Operation to perform: create_videos or create_tackle_videos")
    parser.add_argument("--clip_length", type=int, help="Length of the video clips to be created")
    parser.add_argument("--frame_overlap", type=int, help="Number of frames required to overalp the tackle bounds")#

    args = parser.parse_args()

    match args.op:
        case "count":
            print(get_number_of_tackle_clips(args.in_path))
        case "create_videos":
            video_tackle_data = transform_csv_to_object(args.in_path)
            for dr in ["train", "validation", "test"]:
                with open(os.path.join(args.out_dir, dr, "labels.csv"), 'w', newline='', encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["video_name", "start_frame", "end_frame", "is_tackle"])
            create_videos(args.in_path, args.out_dir, video_tackle_data, args.clip_length, args.frame_overlap)
        case "create_tackles":
            video_tackle_data = transform_csv_to_object(args.in_path)
            create_tackle_videos(args.in_path, args.out_dir, video_tackle_data)

if __name__ == "__main__":
    main()
