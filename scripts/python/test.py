import torch
import sys
import os
import cv2


def main():
    dir_path="/dcs/large/u2102661/CS310/datasets/activity_recogniser_5_frames/original_clips"
    total_duration = 0
    print(os.listdir(dir_path))
    for filename in os.listdir(dir_path):
        if not filename.endswith(".mp4"):
            continue
        print(f"=== {filename} ===")
        cap = cv2.VideoCapture(os.path.join(dir_path, filename))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        total_duration += duration
        cap.release()
    print("Total duration (in seconds):", total_duration)
    print("Total duration (in minutes):", total_duration / 60)

if __name__ == "__main__":
    main()