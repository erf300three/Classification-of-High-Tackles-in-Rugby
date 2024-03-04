import os
import sys
import cv2
import moviepy.editor as moviepy
import csv
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from action_recogniser_model import ActionRecogniserModel, action_prediction
from pose_classifier_model import PoseClassifierModel, tackle_classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Computes the normalised distance between player1's estimated head location and player2's keypoints
def compute_player_distances(player1, player2):
    player2_distances = []
    player1_head_x = 0
    player1_head_y = 0
    player1_divisor = 0
    # The first 5 key points are: nose, left eye, right eye, left ear and right ear which we will use to estimate the head location
    for i in range(5):
        if player1[i][0] != 0:
            player1_head_x += player1[i][0]
            player1_head_y += player1[i][1]
            player1_divisor += 1
    # Computing the average head location of the points that were found
    if player1_divisor != 0:
        player1_head_x /= player1_divisor
        player1_head_y /= player1_divisor
    else:
        # If no head keypoints are found, we will estimate the head location from the other existing key points
        # Its x coordinate will be the average of the left and right shoulder
        if player1[5][0] != 0 and player1[6][0] != 0:
            player1_head_x = player1[5][0] + player1[6][0] / 2
        elif player1[5][0] == 0 and player1[6][0] != 0:
            player1_head_x = player1[6][0]
        elif player1[5][0] != 0 and player1[6][0] == 0:
            player1_head_x = player1[5][0]
        # The hip keypoints must exist if the player is considered to be in the tackle
        else: 
            player1_head_x = player1[11][0] + player1[12][0] / 2

        # Its y coordinate will be the distance between the ankle and shoulder multiplied by 1.08
        # We are checking the left ankle and left shoulder first
        if player1[5][1] != 0 and player1[15][1] != 0:
            player1_head_y = player1[5][1] - player1[15][1] * 1.08
        elif player1[6][1] != 0 and player1[16][1] != 0:
            player1_head_y = player1[6][1] - player1[16][1] * 1.08
        # If these points arent available we will use the left hip and left shoulder multiplied by 2.75
        elif player1[11][1] != 0 and player1[5][1] != 0:
            player1_head_y = player1[5][1] - player1[11][1] * 2.75
        elif player1[12][1] != 0 and player1[6][1] != 0:
            player1_head_y = player1[6][1] - player1[12][1] * 2.75
    
    if (player1_head_x == 0 and player1_head_y == 0):
        print("Not enough keypoints were found to add this to the dataset")
        return []
    # We want to work out the distance between each players keypoints and the average head location of the other player
    for i in range(17):
        if player2[i][0] != 0 and player2[i][1] != 0:
            player2_distances.append([abs(player2[i][0] - player1_head_x), abs(player2[i][1] - player1_head_y)])
        elif player2[i][0] != 0: 
            player2_distances.append([abs(player2[i][0] - player1_head_x), 0])
        elif player2[i][1] != 0:
            player2_distances.append([0, abs(player2[i][1] - player1_head_y)])
        else:
            player2_distances.append([0, 0])
    player2_distances = np.array(player2_distances)
    return player2_distances


def pipeline(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Load the model
    action_model = torch.load("/dcs/large/u2102661/CS310/models/activity_recogniser/r3d_18/run4/best.pt", map_location=device).to(device)

    # Determine if any of the frames are likely to contain a tackle using our action recogniser model
    frames = action_prediction(action_model, video_path, is_full_version=False)

    
    if len(frames) == 0:
        print("No tackles detected")
        return
    
    # We now need to pad out the selected frame by 16 frames either side to ensure that we have enough frames to accurately localise the tackle
    
    buffer = 10
    tackle_clips = []
    for frame in frames:
        temp = []
        for i in range(frame - buffer, frame + buffer):
            temp.append(i)
        tackle_clips.append(temp)


    print(f"Frames containing tackles: {frames}")

    # Documenting the frames that contain tackles by writing them to their own video file using OpenCV  
    for clip in tackle_clips:
        print(f"Clips: {clip}")
        # We want to write the frames to a new video file 
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        clip_name = f"{video_path.split('/')[-1][:-4]}_{clip[0]}_{clip[-1]}.mp4"
        out = cv2.VideoWriter(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_clips/{clip_name}", fourcc, fps, (width, height))

        for frame in clip:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            out.write(img)
        out.release()

        # Conver the video to h264 format using moviepy
        clip = moviepy.VideoFileClip(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_clips/{clip_name}")
        clip.write_videofile(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_clips/{clip_name[:-4]}_h264.mp4", codec="libx264", audio_codec="aac")
        # Delete the original video
        os.remove(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_clips/{clip_name}")
    
    # If we have detected a tackle we will use a custom YOLOv8 model to detect the bounding box of the tackle 
    yolo = YOLO('/dcs/large/u2102661/CS310/models/tackle_location/train/weights/best.pt')
    
    tackle_localisation = {}

    # Loop through all of the frames that we have detected a tackle in and use YOLO to localise the tackle
    for tackle in tackle_clips:
        for frame in tackle:
            # Load the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = yolo.predict(img, save=False, show=False, project="/dcs/large/u2102661/CS310/model_evaluation/pipeline", name="tackle_location")
            # Save the original frame
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if not os.path.exists(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_frames/{video_path.split('/')[-1][:-4]}"):
                os.makedirs(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_frames/{video_path.split('/')[-1][:-4]}")
            cv2.imwrite(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_frames/{video_path.split('/')[-1][:-4]}/frame_{frame}.jpg", img)
            # Plot the results on the image
            annotated_frame = results[0].plot()
            # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            if not os.path.exists(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_location/{video_path.split('/')[-1][:-4]}"):
                os.makedirs(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_location/{video_path.split('/')[-1][:-4]}")
            cv2.imwrite(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_location/{video_path.split('/')[-1][:-4]}/frame_{frame}.jpg", annotated_frame)
            # Save the bounding box results to the dictionary
            # If for some reason the model has detected multiple tackles in a single frame we will store the bounding box with the highest confidence
            best_confidence = 0
            for r in results:
                # Check if there are any bounding boxes
                if len(r.boxes) > 0:
                    if r.boxes[0].conf > best_confidence:
                        best_confidence = r.boxes[0].conf
                        tackle_localisation[frame] = [r.boxes[0].xyxy]

    if len(tackle_localisation) == 0:
        print("No tackles detected")
        # We want to return a path to where the clipped video is stored so that the tackle can be manually reviewed
        return


    # We now want to take bounding boxes and use them to crop the frames to only include that region +- some padding
    for frame in tackle_localisation:
        print(f"Frame: {frame} Bounding box: {tackle_localisation[frame]}")
        temp3 = tackle_localisation[frame][0].to("cpu").numpy()[0]
        # Get each of the coordinates of the bounding box
        x1, y1, x2, y2 = round(temp3[0]), round(temp3[1]), round(temp3[2]), round(temp3[3])
        print(f"x1: {x1} y1: {y1} x2: {x2} y2: {y2}")
        # Padding is 10% of the width and height of the bounding box
        padding_x = int((x2 - x1) * 0.25)
        padding_y = int((y2 - y1) * 0.25)
        # Get the image that we want to crop
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Prevent the padding from going outside of the image
        new_x1 = max(0, int(x1 - padding_x))
        new_x2 = min(img.shape[1], int(x2 + padding_x))
        new_y1 = max(0, int(y1 - padding_y))
        new_y2 = min(img.shape[0], int(y2 + padding_y))
        # Crop the image
        cropped_img = img[new_y1:new_y2, new_x1:new_x2]
        # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        # Save the cropped image
        if not os.path.exists(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_crops/{video_path.split('/')[-1][:-4]}"):
            os.makedirs(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_crops/{video_path.split('/')[-1][:-4]}")
        cv2.imwrite(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_crops/{video_path.split('/')[-1][:-4]}/frame_{frame}.jpg", cropped_img)

            
    # Apply pose estimiation to the cropped images to get the poses of the players involved in the tackle
    for frame in os.listdir(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_frames/{video_path.split('/')[-1][:-4]}"):
        print(frame)
        # Load the frame
        
        yolo = YOLO('yolov8x-pose.pt')
        results = yolo.predict(
            f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_frames/{video_path.split('/')[-1][:-4]}/{frame}",
            save=False,
            show=False,
            imgsz=640
        )
        annotated_frame = results[0].plot()
        # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        if not os.path.exists(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/pose_estimation/{video_path.split('/')[-1][:-4]}"):
            os.makedirs(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/pose_estimation/{video_path.split('/')[-1][:-4]}")
        cv2.imwrite(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/pose_estimation/{video_path.split('/')[-1][:-4]}/{frame}", annotated_frame)
        for r in results:
            # We want to output the pose estimations have their waist or shoulder keypoints within the bounding box of the tackle
            # Check if there is a bounding box for the frame
            # Get the frame number of the tackle frame of the form Team1_vs_Team2_frame_XXX.
            frame_number = int(frame.split("_")[1][:-4])


            if frame_number in tackle_localisation:
                # Get the coordinates of the bounding box
                temp3 = tackle_localisation[frame_number][0].to("cpu").numpy()[0]
                x1, y1, x2, y2 = round(temp3[0]), round(temp3[1]), round(temp3[2]), round(temp3[3])
                
                # # Compare to the shape of empty keypoints
                if r.keypoints[0].xy.to("cpu").shape == (1, 0, 2):
                    continue

                players_in_tackle = []
                players_in_tackle_normalised = []
                players_in_tackle_confidence = []

                pose_classification_model = torch.load("/dcs/large/u2102661/CS310/models/pose_estimation/run1/last.pt", map_location=device).to(device)
                for player_keypoint in r.keypoints:
                    points = player_keypoint.xy.to("cpu").numpy()[0]
                    points_normalised = player_keypoint.xyn.to("cpu").numpy()[0]
                    confidence = player_keypoint.conf.to("cpu").numpy()
                    print("points: ", points)
                    # Check if the hip or shoulder keypoints are within the bounding 
                    # The hip shoulder points are index 11 and 12 and the shoulder points are index 5 and 6
                    left_hip = points[11]
                    right_hip = points[12]
                    left_shoulder = points[5]
                    right_shoulder = points[6]
                    # Check if the hip or shoulder keypoints are within the bounding box
                    points_to_check = [left_hip, right_hip, left_shoulder, right_shoulder]
                    or_gate = False
                    for point in points_to_check:
                        if x1 < point[0] < x2 and y1 < point[1] < y2:
                            print(f"Point: {point} is within the bounding box")
                            or_gate = True
                        else:
                            print(f"Point: {point} is not within the bounding box")
                    if or_gate:
                        print("Player is within the bounding box")
                        players_in_tackle.append(points)
                        players_in_tackle_normalised.append(points_normalised)
                        players_in_tackle_confidence.append(confidence)
                        

                    if len(players_in_tackle) < 2:
                        print("Not enough players in the tackle")
                        continue
                        
                    for play1 in range(len(players_in_tackle)):
                        for play2 in range(play1 + 1, len(players_in_tackle)):
                            player1 = players_in_tackle[play1]
                            player2 = players_in_tackle[play2]
                            player1_normalised = players_in_tackle_normalised[play1]
                            player2_normalised = players_in_tackle_normalised[play2]
                            player1_confidence = players_in_tackle_confidence[play1]
                            player2_confidence = players_in_tackle_confidence[play2]

                            img = cv2.imread(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/tackle_frames/{video_path.split('/')[-1][:-4]}/{frame}")

                            player1 = np.insert(player1, 2, player1_confidence, axis=1)
                            player2 = np.insert(player2, 2, player2_confidence, axis=1)
                            annotator = Annotator(img, line_width=2, font_size=1.0)

                            # Draw the annotations
                            annotator.kpts(player1)
                            annotator.kpts(player2)
                            out_img = annotator.result()

                            player1_distances = compute_player_distances(player1_normalised, player2_normalised)
                            player2_distances = compute_player_distances(player2_normalised, player1_normalised)


                            # We want to classify the pose of the players in the tackle
                            predicted_class = tackle_classification(pose_classification_model, player1_distances, player2_distances, device)
                            # We want to write the classification onto the image
                            # This is in BGR format
                            if predicted_class == 0:
                                cv2.putText(out_img, "No Tackle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            elif predicted_class == 1:
                                cv2.putText(out_img, "Low Tackle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            elif predicted_class == 2:
                                cv2.putText(out_img, "High Tackle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            # Save the image
                            if not os.path.exists(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/pose_classification/{video_path.split('/')[-1][:-4]}"):
                                os.makedirs(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/pose_classification/{video_path.split('/')[-1][:-4]}")#
                            cv2.imwrite(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/pose_classification/{video_path.split('/')[-1][:-4]}/{frame[:-4]}_player{play1}_player{play2}.jpg", out_img)





            

                    





    # We now want to take these cropped images and use them to complete pose estimation on to get the poses of the players involved in the tackle
    


def main():
    dir_path = "/dcs/large/u2102661/CS310/datasets/negative_clips"
    # dir_path = "/dcs/large/u2102661/CS310/datasets/anomaly_detection/original_videos"
    for file in os.listdir(dir_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(dir_path, file)
            pipeline(video_path)
    # pipeline("/dcs/large/u2102661/CS310/datasets/activity_recogniser/original_clips/Darlington_V_Sedgley_Yellow_High_Tackle1.mp4")

if __name__ == "__main__":
    main()
