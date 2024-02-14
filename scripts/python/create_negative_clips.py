import os 
import sys
import cv2
import csv
import numpy as np
import pandas as pd
import random
import argparse


def create_negative_clips(dir_path, output_path, clip_length, num_clips):
    # Create a list of all the video files in the directory
    video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]#
    print("video_files: ", video_files) 

    # For each of the videos we want to create 30 second clips from the videos of random starting points
    with open(os.path.join(dir_path, 'negative_clips.csv'), 'w', newline='') as file:
        csv_file = csv.DictWriter(file, fieldnames=['video', 'start_frame', 'end_frame'])
        csv_file.writeheader()
        for video in video_files: 
            print("=====================================" + video + "=====================================")
            cap = cv2.VideoCapture(os.path.join(dir_path, video))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("fps: ", fps)
            print("total_frames: ", total_frames)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            for i in range(num_clips):
                print("-------------------------------------" + str(i + 1) + "-------------------------------------")
                start_frame = random.randint(0, total_frames - (clip_length * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                print("start_frame: ", start_frame)
                frames = []
                for j in range(clip_length * int(fps)):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        break
                print("frames: ", len(frames))  
                print("clip_length: ", clip_length * int(fps))
                if len(frames) == clip_length * int(fps):
                    out = cv2.VideoWriter(os.path.join(output_path, (video[:-4].replace(" ", "_")) + '_' + str(i+ 1) + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    csv_file.writerow({'video': video, 'start_frame': start_frame, 'end_frame': start_frame + clip_length * int(fps)})
                # os.system(f'ffmpeg -i {os.path.join(output_path, (video[:-4].replace(" ", "_")) + "_" + str(i + 1) + ".mp4")} -vcodec libx264 {os.path.join(output_path, (video[:-4].replace(" ", "_")) + "_" + str(i + 1) + "_compressed.mp4")}')
            cap.release()
        file.close()

def main():

    parser  = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Path to the directory containing the videos")
    parser.add_argument("--out", help="Path to the directory to save the negative clips")
    args = parser.parse_args()

    dir_path = args.dir
    output_path = args.out
    clip_length = 30 # seconds
    num_clips = 30
    create_negative_clips(dir_path, output_path, clip_length, num_clips)

if __name__ == "__main__":
    main()
    


            