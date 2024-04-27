import os 
import sys
import csv
import cv2
import random

def transform_csv_to_object(in_path):
    output_file = os.path.join(in_path, "tackles.csv")
    fields = ["video_name", "start_frame", "end_frame"]
    data = {}
    with open(output_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["video_name"] not in data:
                data[row["video_name"]] = []
            data[row["video_name"]].append((int(row["start_frame"]), int(row["end_frame"])))
    return data

def create_tackle_videos(in_path, out_dir, data):
    for video in os.listdir(in_path):
        print(f"===================={video}====================")
        if not(video.endswith(".mp4")):
            continue
        video_path = os.path.join(in_path, video)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        list_of_tackles = data[video]
        if not os.path.exists(os.path.join(out_dir, "tackles")):
            os.makedirs(os.path.join(out_dir, "tackles"))
        for idx, (tackle_start, tackle_end) in enumerate(list_of_tackles):
            outpath = os.path.join(out_dir, "tackles", video[:-4] + "_" + str(idx) + ".mp4")
            print(outpath)
            writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, tackle_start - (fps * 3)))
            tackle_end = min(tackle_end + (fps * 2), cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
def create_videos(in_path, out_dir, data):
    # Define the length of the clips and the require number of frames to overlap the tackle frames
    clip_length = 5
    frame_overlap = 3
    for video in os.listdir(in_path):
        print(video)
        if not(video.endswith(".mp4")):
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
            with open(csv_path, 'a', newline='') as csvfile:
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
    Input: in_path - path to the dataset
    Output: number_of_tackle_clips - number of tackle clips in the dataset
            number_of_non_tackle_clips - number of non-tackle clips in the dataset
            total_number_of_clips - total number of clips in the dataset
    """ 
    number_of_tackle_clips = 0 
    total_number_of_clips = 0

    for dir in ["validation"]:
        with open(os.path.join(in_path, dir, "labels.csv"), 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_number_of_clips += 1
                if row["is_tackle"] == "1":
                    number_of_tackle_clips += 1

    return number_of_tackle_clips, total_number_of_clips - number_of_tackle_clips, total_number_of_clips
            
# Location   Number of tackle clips    Number of non-tackle clips    Total number of clips
# All dirs:  570                       11337                         11907
# Train:     442                       9027                          9469
# Test:      72                        1143                          1215
# Validation 56                        1167                          1223
def main(): 
    in_path = "/dcs/large/u2102661/CS310/datasets/activity_recogniser/original_clips"
    out_dir = "/dcs/large/u2102661/CS310/datasets/activity_recogniser_5_frames"
    # video_tackle_data = transform_csv_to_object(in_path)

    # with open(os.path.join(out_dir, "train", "labels.csv"), 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["video_name", "start_frame", "end_frame", "is_tackle"])#
    # with open(os.path.join(out_dir, "validation", "labels.csv"), 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["video_name", "start_frame", "end_frame", "is_tackle"])
    # with open(os.path.join(out_dir, "test", "labels.csv"), 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["video_name", "start_frame", "end_frame", "is_tackle"])

    # create_videos(in_path, out_dir, video_tackle_data)

    print(get_number_of_tackle_clips(out_dir))
    # create_tackle_videos(in_path, out_dir, video_tackle_data)

if __name__ == "__main__":
    main()