import os 
import sys
import cv2 
import csv
import math
import argparse

def view_frames(in_path):
    output_file = os.path.join(in_path, "tackles.csv")
    fields = ["video_name", "start_frame", "end_frame"]
    data = []
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for file in os.listdir(in_path):
            if file.endswith(".mp4"):
                print("=========================================" + file + "=========================================")
                vidcap = cv2.VideoCapture(os.path.join(in_path, file))
                print("Fps: " + str(vidcap.get(cv2.CAP_PROP_FPS)))
                frame_number = 0
                list_of_tackle_frames = []
                while vidcap.isOpened():
                    ret, frame = vidcap.read()
                    print("Frame: " + str(frame_number))
                    if ret == False:
                        break
                    cv2.imshow('Frame' , frame)
                    key_pressed = cv2.waitKey(0) & 0xFF
                    if key_pressed == ord("q"):
                        break
                    elif key_pressed == ord("c"):
                        frame_number += 1
                        continue
                    elif key_pressed == ord("s"):
                        list_of_tackle_frames.append(frame_number)
                        frame_number += 1
                    elif key_pressed == ord("m"):
                        if (len(list_of_tackle_frames) > 0):
                            print("Writing to file")
                            start_tackle_frame = list_of_tackle_frames[0]
                            end_tackle_frame = list_of_tackle_frames[-1]
                            writer.writerow({"video_name": file, "start_frame": start_tackle_frame, "end_frame": end_tackle_frame})
                            list_of_tackle_frames.clear()
                if (len(list_of_tackle_frames) > 0):
                    print("Writing to file")
                    start_tackle_frame = list_of_tackle_frames[0]
                    end_tackle_frame = list_of_tackle_frames[-1]
                    writer.writerow({"video_name": file, "start_frame": start_tackle_frame, "end_frame": end_tackle_frame})
                vidcap.release()
                cv2.destroyAllWindows()


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
  

"""
    This function will take the data object and the different videos and iterate through them to create anomaly videos 
    and normal videos. If the anomaly occurs before the first 30 frames then we will not create a normal video before 
    the anomaly occurs. If the video has multiple anomalies then we will create multiple anomaly videos and normal videos.
"""
def write_new_videos(in_path, out_path, data):
    for file in os.listdir(in_path):
        if file.endswith(".mp4"):
            anomalies = data[file]
            anomalies.sort(key=lambda x: x[0])
            duration_between_anomalies = [anomalies[i+1][0] - anomalies[i][0] for i in range(len(anomalies) - 1)]
            
            print("=========================================" + file + "=========================================")
            
            # Calculate the tackle bounds
            tackle_bounds = []
            for anomaly in anomalies:
                tackle_bounds.append((anomaly[0] - 3, anomaly[1] + 3))
            
            print("Tackle bounds: ", tackle_bounds)

            # Calculate the anomaly bounds 
            anomaly_bounds = []
            for i in range(len(anomalies)):
                if i == 0 and len(duration_between_anomalies) > 0:
                    anomaly_bounds.append((0, anomalies[i][1] +  math.floor(duration_between_anomalies[i] / 2)))
                elif i == 0 and len(duration_between_anomalies) == 0:
                    anomaly_bounds.append((0, sys.maxsize))
                elif i == len(anomalies) - 1:
                    anomaly_bounds.append((anomalies[i][0] - math.ceil(duration_between_anomalies[i-1] / 2) + 1, sys.maxsize))
                else:
                    anomaly_bounds.append((anomalies[i][0] - math.ceil(duration_between_anomalies[i-1] / 2) + 1, anomalies[i][1] + math.floor(duration_between_anomalies[i] / 2)))

            print("Anomaly bounds: ", anomaly_bounds)

            # Calculate the normal bounds
            normal_bounds = []
            i = 0
            for i in range(len(anomalies)):
                if i == 0 and anomalies[i][0] > 30:
                    normal_bounds.append((0, anomalies[i][0] - 4))
                elif i > 0 and duration_between_anomalies[i-1] > 100:
                    normal_bounds.append((anomalies[i-1][1] + 4, anomalies[i][0] - 4))
            normal_bounds.append((anomalies[i][1] + 4, sys.maxsize))
            print("Normal bounds: ", normal_bounds)


            cap = cv2.VideoCapture(os.path.join(in_path, file))

            # Write the normal videos
            for idx, (start_frame, end_frame) in enumerate(normal_bounds):
                file_out = os.path.join(out_path,"normal_videos" , file[:-4] + "_normal_" + str(idx) + ".mp4")
                writer = cv2.VideoWriter(file_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
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

            # Write the anomaly videos
            for idx, (start_frame, end_frame) in enumerate(anomaly_bounds):
                file_out = os.path.join(out_path,"anomaly_videos" , file[:-4] + "_anomaly_" + str(idx) + ".mp4")
                writer = cv2.VideoWriter(file_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
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
            
            # Write the tackle videos
            for idx, (start_frame, end_frame) in enumerate(tackle_bounds):
                file_out = os.path.join(out_path,"tackle_videos" , file[:-4] + "_tackle_" + str(idx) + ".mp4")
                writer = cv2.VideoWriter(file_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
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


def view_video(in_path):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imshow('Frame' , frame)
        key_pressed = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key_pressed == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()          
                    


def main():
    parser = argparse.ArgumentParser(description='Manipulate data')
    parser.add_argument("--op", type=str, default="view_frames")
    parser.add_argument("--in_path", type=str, default="")
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



# Modifications made to the csv file directly
#  Birmingham_v_Rosslyn_Yellow_High_Tackle1.mp4 need to bring first tackle up by 20 frames 
#  Blackheath_v_Rams_Yellow_High_Tackle1.mp4 first one is an error 
#  Bury_St_Edmonds_v_Sevenoaks_Yellow_High_Tackle1.mp4 starts with a tackle change to 0 frame
#  Bury_v_Barnes_Yellow_High_Tackle2.mp4 take back 15 frames and lamppost is in the way
#  Bury_v_North_Walsham_Yellow_Card_HIgh_Tackle1.mp4 extend the end by 10 frames
#  Clifton_v_Newport_Yellow_High_Tackle2.mp4 3rd tackle bring up by 5 frames
#  Dorking_v_Walsham_Yellow_3_High_Tackle1.mp4 first tackle needs to be brought forward by 15 frames
#  Dorking_v_Walsham_Yellow_High_Tackle1.mp4 first tackle is error 
#  Dorking_v_Worthing_Yellow_Card_High_Tackle1.mp4 first tackle is error
#  Exeter_Uni_v_Dings_Yellow_High_Tackle2.mp4 trim last tackle by 25 frames
#  Henley_v_Old_Albanians_Yellow_High_Tackle1.mp4 second tackle is error
#  Hull_v_Billingham_Yellow_High_Tackle_21.mp4 first tackle is error
#  Leeds_v_Hull_Yellow_High_Tackle1.mp4 first tackle broguht forward by 10 frames
#  Rosslyn_Park_v_Plymouth_Albion_Yellow_High_Tackle2.mp4 bring first tackle forward by 10 frames
#  Wimbledon_v_Bury_Red_High_Tackle1.mp4 first tackle is error


# Need to rewatch
#  Blackheath_v_Richmond_Yellow_High_Tackle1.mp4 (everything is too far away)
#  Bury_v_Barnes_Red_High_Tackle1.mp4 (Late tackle on the kicker)
#  Dings_v_Dudley_Yellow_HIgh_Tackle_off_the_ball1.mp4
#  Dorking_v_Westcombe_Park_Yellow_High_late_shoulder.1.mp4 (late tackle)
#  Fylde_v_Huddersfield_Red_High_Tackle1.mp4 (cant see anything plus there is a punch upc)
#  Guernsey_v_Tonbridge_Yellow_High_Tackle1.mp4
#  Huddersfield_v_Lymm_Yellow_High_Tackle1.mp4
#  Hull_v_Billingham_Yellow_High_Tackle_21.mp4
#  Rosslyn_Park_v_Bishops_Stortford_Yellow_High_Tackle1.mp4
#  Sevenoaks_v_Canterbury_Yellow_High_Tackle4.mp4
#  Tonbridge_v_Bury_St_Edmonds_Yellow_High_Tackle1.mp4
#  Wharfedale_v_Flyde_Yellow_High_Tackle1.mp4

    
# Need to remove 
# Exeter_v_Horneys_Yellow_Card_High_Tackle1.mp4 is not caught by the camera (players in the way)
# Old_Albanians_v_Dorking_Yellow_High_Tackle1.mp4 (no tackle visible)

