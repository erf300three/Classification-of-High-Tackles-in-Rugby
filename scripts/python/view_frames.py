import os 
import sys
import csv
import math
import argparse
import cv2

def view_frames(in_path):
    """
    This function will take the path to a directory and allows the user to view the frames of the
    all the videos. These frames are viewed sequentially allowing the user to record whether the 
    frame contained a tackle or not. The user can press the following keys:
        - q: Quit the video
        - c: Continue to the next frame
        - s: Save the current frame as the start of the tackle
        - m: Signify the end of the tackle and write the start and end frames to the csv file

    Input:
        in_path: The path to the directory containing the videos to label
    Output:
        None
    """
    output_file = os.path.join(in_path, "tackles.csv")
    fields = ["video_name", "start_frame", "end_frame"]
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for file in os.listdir(in_path):
            if file.endswith(".mp4"):
                print("================" + file + "================")
                vidcap = cv2.VideoCapture(os.path.join(in_path, file))
                print("Fps: " + str(vidcap.get(cv2.CAP_PROP_FPS)))
                frame_number = 0
                list_of_tackle_frames = []
                while vidcap.isOpened():
                    ret, frame = vidcap.read()
                    print("Frame: " + str(frame_number))
                    if not ret:
                        break
                    cv2.imshow('Frame' , frame)
                    key_pressed = cv2.waitKey(0) & 0xFF
                    if key_pressed == ord("q"):
                        break
                    if key_pressed == ord("c"):
                        frame_number += 1
                        continue
                    if key_pressed == ord("s"):
                        list_of_tackle_frames.append(frame_number)
                        frame_number += 1
                    elif key_pressed == ord("m"):
                        if len(list_of_tackle_frames) > 0:
                            print("Writing to file")
                            start_tackle_frame = list_of_tackle_frames[0]
                            end_tackle_frame = list_of_tackle_frames[-1]
                            writer.writerow({
                                "video_name": file,
                                "start_frame": start_tackle_frame, 
                                "end_frame": end_tackle_frame
                            })
                            list_of_tackle_frames.clear()
                if len(list_of_tackle_frames) > 0:
                    print("Writing to file")
                    start_tackle_frame = list_of_tackle_frames[0]
                    end_tackle_frame = list_of_tackle_frames[-1]
                    writer.writerow({
                        "video_name": file,
                        "start_frame": start_tackle_frame,
                        "end_frame": end_tackle_frame
                    })
                vidcap.release()
                cv2.destroyAllWindows()


def transform_csv_to_object(in_path):
    """
    This function will take the csv file and transform it into a dictionary object where the key is 
    the video name and the value is a list of tuples where the tuple represents the start and end 
    frame of the tackle.
    
    Input: 
        in_path: The path to the csv file
    Output:
        data: A dictionary object where the key is the video name and the value is a list of tuples 
              where the tuple represents the start and end frame of the tackle.
    """
    output_file = os.path.join(in_path, "tackles.csv")
    data = {}
    with open(output_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["video_name"] not in data:
                data[row["video_name"]] = []
            data[row["video_name"]].append((int(row["start_frame"]), int(row["end_frame"])))
    return data

def write_video(in_path, out_path, data):
    """
    This function will take the bounds of a video and write a new video with the frames between the
    start and end frame. The new video will be saved in the out_path directory.

    Input:
        in_path: The path to the video
        out_path: The path where the new video will be saved
        data: A tuple object where the first element is the start frame and the second element is
                the end frame.
    Output:
        A new video which will be saved at the out_path
    """
    (start_frame, end_frame) = data
    cap = cv2.VideoCapture(os.path.join(in_path))
    video_settings = (cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
    writer = cv2.VideoWriter(out_path, *video_settings)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret = True
    while cap.isOpened() and ret and writer.isOpened():
        ret, frame = cap.read()
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_number <= end_frame:
            writer.write(frame)
        if frame_number > end_frame:
            break
    writer.release()

def write_new_videos(in_path, out_path, data):
    """
    This function will take the data object created from the tackle bounds and iterate through them
    to create new videos for the anomaly detection model. The new videos will be created in the
    following way:
        - Anomaly videos: These videos will contain frames which include the tackle 
        - Normal videos: These videos will contain frames which do not include the tackle
        - Tackle videos: These videos will only contain the frames which include the tackle
    If the anomaly occurs before the first 30 frames then we will not create a normal video before
    the anomaly occurs. If the video has multiple anomalies then we will create multiple anomaly
    videos and normal videos.

    Input:
        in_path: The path to the directory containing the videos
        out_path: The path to the directory where the new videos will be saved
        data: A dictionary object where the key is the video name and the value is a list of tuples
                where the tuple represents the start and end frame of the tackle.
    Output:
        All videos will be saved in the out_path directory
    """
    for file in os.listdir(in_path):
        if not file.endswith(".mp4"):
            continue
        anomalies = data[file]
        anomalies.sort(key=lambda x: x[0])
        duration_between_anomalies = [anomalies[i+1][0] - anomalies[i][0] for i in range(len(anomalies) - 1)]
        print("================" + file + "================")
        
        # Calculate the tackle bounds
        tackle_bounds = []
        for anomaly in anomalies:
            tackle_bounds.append((anomaly[0] - 3, anomaly[1] + 3))
        print("Tackle bounds: ", tackle_bounds)

        # Calculate the anomaly bounds
        anomaly_bounds = []
        for idx, anomaly in enumerate(anomalies):
            if idx == 0 and len(duration_between_anomalies) > 0:
                anomaly_bounds.append((
                    0,                                                                  # start
                    anomalies[idx][1] + math.floor(duration_between_anomalies[idx] / 2) # end 
                ))
            elif idx == 0 and len(duration_between_anomalies) == 0:
                anomaly_bounds.append((
                    0,
                    sys.maxsize
                ))
            elif idx == len(anomalies) - 1:
                anomaly_bounds.append((
                    anomalies[idx][0] - math.ceil(duration_between_anomalies[idx-1] / 2) + 1,
                    sys.maxsize
                ))
            else:
                anomaly_bounds.append((
                    anomalies[idx][0] - math.ceil(duration_between_anomalies[idx-1] / 2) + 1,
                    anomalies[idx][1] + math.floor(duration_between_anomalies[idx] / 2)
                ))
        print("Anomaly bounds: ", anomaly_bounds)

        # Calculate the normal bounds
        normal_bounds = []
        i = 0
        for idx, anomaly in enumerate(anomalies):
            if idx == 0 and anomalies[idx][0] > 30:
                normal_bounds.append((0, anomalies[idx][0] - 4))
            elif idx > 0 and duration_between_anomalies[idx-1] > 100:
                normal_bounds.append((anomalies[idx-1][1] + 4, anomalies[idx][0] - 4))
        normal_bounds.append((anomalies[i][1] + 4, sys.maxsize))
        print("Normal bounds: ", normal_bounds)
        
        # Create the directories
        if not os.path.exists(os.path.join(out_path, "normal_videos")):
            os.makedirs(os.path.join(out_path, "normal_videos"))
        if not os.path.exists(os.path.join(out_path, "anomaly_videos")):
            os.makedirs(os.path.join(out_path, "anomaly_videos"))
        if not os.path.exists(os.path.join(out_path, "tackle_videos")):
            os.makedirs(os.path.join(out_path, "tackle_videos"))

        # Write the normal videos
        for idx, (start_frame, end_frame) in enumerate(normal_bounds):
            file_out = os.path.join(out_path,"normal_videos", f"{file[:-4]}_normal_{str(id)}.mp4")
            write_video(os.path.join(in_path, file), file_out, (start_frame, end_frame))
        # Write the anomaly videos
        for idx, (start_frame, end_frame) in enumerate(anomaly_bounds):
            file_out = os.path.join(out_path,"anomaly_videos", f"{file[:-4]}_anomaly_{str(id)}.mp4")
            write_video(os.path.join(in_path, file), file_out, (start_frame, end_frame))
        # Write the tackle videos
        for idx, (start_frame, end_frame) in enumerate(tackle_bounds):
            file_out = os.path.join(out_path,"tackle_videos", f"{file[:-4]}_tackle_{str(id)}.mp4")
            write_video(os.path.join(in_path, file), file_out, (start_frame, end_frame))

def view_video(in_path):
    """
    This function will take the path to a video and allow the user to view the video. 
    The user can stop watching the video by pressing the "q" key.

    Input:
        in_path: The path to the video
    """
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame' , frame)
        key_pressed = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key_pressed == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    This function will parse the arguments and call the appropriate function based on the operation
    that the user wants to perform.
    """
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Manipulate data')
    parser.add_argument("--op", type=str, default="view_frames", required=True)
    parser.add_argument("--in_path", type=str, default="", required=True)
    parser.add_argument("--out_path", type=str, default="")
    args = parser.parse_args()
    match args.op:
        case "view_frames":
            for file in os.listdir(args.in_path):
                if file.endswith(".mp4"):
                    view_video(os.path.join(args.in_path, file))
        case "write_new_videos":
            data = transform_csv_to_object(args.in_path)
            write_new_videos(args.in_path, args.out_path, data)
        case "view_frames":
            view_frames(args.in_path)

if __name__ == "__main__":
    main()
