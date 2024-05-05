"""
This file contains all of the code requried to run the pipeline. The pipeline is used to detect tackles in rugby videos 
and classify the legality of the tackles based on the domestic law variation. All of the functions within this 
file are used during the pipeline and should not be called individually. The pipeline is called by the main.py file
and is used to process the all video files in the dataset. The pipeline will output the results of the pipeline to a 
results.csv file which will be saved in the output directory. The results.csv file will contain the following columns:
- Video Name: The name of the video file
- Tackle Number: The number of the tackle in the video
- Tackle Frames: The frame number of the first frame in the tackle clip generated
- Tackles Located: The number of tackles that were located in the video
- People in Tackle: The number of people in the tackle
- Tackle Classification: The classification of the tackle
- Ground Truth: The ground truth classification of the tackle
- Ground Truth Tackle Start Frames: The frame number of the start of the ground truth tackle
- Ground Truth Tackle End Frames: The frame number of the end of the ground truth tackle
"""
import os
import csv
import argparse
import cv2
import moviepy.editor as moviepy
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
# We still need to import the models even though they are not directly called in this file
from action_recogniser_model import ActionRecogniserModel, action_prediction
from pose_classifier_model import PoseClassifierModel, tackle_classification

def compute_player_distances(player1, player2):
    """
    This function computes the normalised distance between player1's estimated head location and player2's keypoints.
    The function first estimates the head location of player1. This is completed by taking the average of the 
    available keypoints in the face region. If the face region keypoints are not available then the function will use 
    the distances between available keypoints to estimate the head location through scaling. The function will then
    compute the normalised distance between player1's estimated head location and player2's keypoints.

    Inputs:
        player1: The keypoints of player1
        player2: The keypoints of player2
    
    Outputs:
        player1_head_x: The x coordinate of player1's estimated head location
        player1_head_y: The y coordinate of player1's estimated head location
        player2_distances: The normalised distances between player1's estimated head location and player2's keypoints
        player1_distances: The normalised distances between player2's estimated head location and player1's keypoints
    """
    player2_distances = []
    player1_head_x = 0
    player1_head_y = 0
    player1_divisor = 0

    # We want to determine if the player is standing up or lying down as this will affect the head location
    player1_tallest_keypoint = 0
    player1_shortest_keypoint = 0
    # Find the first index that is not 0
    for i in range(17):
        if player1[i][1] != 0:
            player1_tallest_keypoint = i
            break
    # Find the last index that is not 0
    for i in range(16, -1, -1):
        if player1[i][1] != 0:
            player1_shortest_keypoint = i
            break

    player_facing = ""
    player1_standing = True
    # The height of the player is the distance between the highest and lowest keypoints
    player1_height_y = abs(player1[player1_tallest_keypoint][1] - player1[player1_shortest_keypoint][1])
    player1_height_x = abs(player1[player1_tallest_keypoint][0] - player1[player1_shortest_keypoint][0])

    # Determine if the player is standing up or lying down
    if player1_height_y > player1_height_x:
        player1_standing = True
        if player1[player1_tallest_keypoint][1] < player1[player1_shortest_keypoint][1]:
            player_facing = "up"
        else:
            player_facing = "down"
    else:
        player1_standing = False
        if player1[player1_tallest_keypoint][0] < player1[player1_shortest_keypoint][0]:
            player_facing = "left"
        else:
            player_facing = "right"

    # The first 5 key points are: nose, left eye, right eye, left ear and right ear which we will use to estimate
    # the head location
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
        # Use whether the player is standing up or lying down to determine the head location
        distance = 0
        key_point = 0
        if player1_standing:
            # The x coordinate will be the average of the left and right shoulder
            if player1[5][0] != 0 and player1[6][0] != 0:
                player1_head_x = (player1[5][0] + player1[6][0]) / 2
            elif player1[5][0] == 0 and player1[6][0] != 0:
                player1_head_x = player1[6][0]
            elif player1[5][0] != 0 and player1[6][0] == 0:
                player1_head_x = player1[5][0]
            # The hip keypoints must exist if the player is considered to be in the tackle
            else:
                if player1[11][0] != 0 and player1[12][0] != 0:
                    player1_head_x = (player1[11][0] + player1[12][0]) / 2
                elif player1[11][0] == 0 and player1[12][0] != 0:
                    player1_head_x = player1[12][0]
                elif player1[11][0] != 0 and player1[12][0] == 0:
                    player1_head_x = player1[11][0]
                else:
                    print("Not enough keypoints to determine head location")
                    return 0, 0, np.array([])

            # The y coordinate will be the distance between the ankle and the shoulder multiplied by 1.15 if they exist
            if player1[5][1] != 0 and player1[15][1] != 0:
                distance = abs(player1[5][1] - player1[15][1]) * 1.15
                key_point = player1[15][1]
            elif player1[6][1] != 0 and player1[16][1] != 0:
                distance = abs(player1[6][1] - player1[16][1]) * 1.15
                key_point = player1[16][1]
            # If these points arent available we will use the hips and shoulders multiplied by 1.4
            elif player1[11][1] != 0 and player1[5][1] != 0:
                distance = abs(player1[11][1] - player1[5][1]) * 1.4
                key_point = player1[11][1]
            elif player1[12][1] != 0 and player1[6][1] != 0:
                distance = abs(player1[12][1] - player1[6][1]) * 1.4
                key_point = player1[12][1]
            # If these points arent available we will use the knees and hips multiplied by 2.58
            elif player1[13][1] != 0 and player1[11][1] != 0:
                distance = abs(player1[13][1] - player1[11][1]) * 2.58
                key_point = player1[13][1]
            elif player1[14][1] != 0 and player1[12][1] != 0:
                distance = abs(player1[14][1] - player1[12][1]) * 2.58
                key_point = player1[14][1]
            else:
                print("Not enough keypoints to determine head location")
                return 0, 0, np.array([])

            if player_facing == "up":
                player1_head_y = key_point - distance
            elif player_facing == "down":
                player1_head_y = key_point + distance
        else:
            # The y coordinate will be the average of the left and right shoulder
            if player1[5][1] != 0 and player1[6][1] != 0:
                player1_head_y = (player1[5][1] + player1[6][1]) / 2
            elif player1[5][1] == 0 and player1[6][1] != 0:
                player1_head_y = player1[6][1]
            elif player1[5][1] != 0 and player1[6][1] == 0:
                player1_head_y = player1[5][1]
            # The hip keypoints must exist if the player is considered to be in the tackle
            else:
                if player1[11][1] != 0 and player1[12][1] != 0:
                    player1_head_y = (player1[11][1] + player1[12][1]) / 2
                elif player1[11][1] == 0 and player1[12][1] != 0:
                    player1_head_y = player1[12][1]
                elif player1[11][1] != 0 and player1[12][1] == 0:
                    player1_head_y = player1[11][1]
                else:
                    print("Not enough keypoints to determine head location")
                    return 0, 0, np.array([])

            # The x coordinate will be the distance between the ankle and shoulder multiplied by 1.15 if they exist
            if player1[5][0] != 0 and player1[15][0] != 0:
                distance = abs(player1[5][0] - player1[15][0]) * 1.15
                key_point = player1[15][0]
            elif player1[6][0] != 0 and player1[16][0] != 0:
                distance = abs(player1[6][0] - player1[16][0]) * 1.15
                key_point = player1[16][0]
            # If these points arent available we will use the hips and shoulders multiplied by 1.4
            elif player1[11][0] != 0 and player1[5][0] != 0:
                distance = abs(player1[11][0] - player1[5][0]) * 1.4
                key_point = player1[11][0]
            elif player1[12][0] != 0 and player1[6][0] != 0:
                distance = abs(player1[12][0] - player1[6][0]) * 1.4
                key_point = player1[12][0]
            # If these points arent available we will use the knees and hips multiplied by 2.58
            elif player1[13][0] != 0 and player1[11][0] != 0:
                distance = abs(player1[13][0] - player1[11][0]) * 2.58
                key_point = player1[13][0]
            elif player1[14][0] != 0 and player1[12][0] != 0:
                distance = abs(player1[14][0] - player1[12][0]) * 2.58
                key_point = player1[14][0]
            else:
                print("Not enough keypoints to determine head location")
                return 0, 0, np.array([])

            if player_facing == "left":
                player1_head_x = key_point - distance
            elif player_facing == "right":
                player1_head_x = key_point + distance

    if (player1_head_x == 0 or player1_head_y == 0):
        print("Not enough keypoints were found to add this to the dataset")
        return 0, 0, np.array([])
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
    return player1_head_x, player1_head_y, player2_distances

def create_buffered_clips(frames, buffer_before, buffer_after, max_frames):
    """
    This function will create a buffered tackle clip from the single frame number that was selected by the Tackle
    Temporal Locator model (action recogniser). The size of the clip is determined by the buffer before and after
    parameters. The function will return a list of frames that comprise the buffered tackle clip.

    Inputs:
        frames: The list of frames that contain a tackle
        buffer_before: The number of frames before the selected frame to include in the clip
        buffer_after: The number of frames after the selected frame to include in the clip
        max_frames: The maximum number of frames in the video
    
    Outputs:
        tackle_clips: A list of frames that comprise the buffered tackle clip
    """
    tackle_clips = []
    for frame in frames:
        start = max(0, frame - buffer_before)
        end = min(max_frames, frame + buffer_after + 1)
        clip = [i for i in range(start, end)]
        tackle_clips.append(clip)
    return tackle_clips

def write_tackle_clips(frames, video_name, video_path, output_path):
    """
    This function will write the list of frames for the tackle clip into a new video file. The function will
    first create a new video file using OpenCV and then convert the video file to h264 format using moviepy.

    Inputs:
        frames: The list of frames that contain a tackle
        video_name: The name of the video file
        video_path: The path to the video file
        output_path: The path to the output directory
    
    Outputs:
        A new video file containing the frames of the tackle clip in h264 format
    """
    cap = cv2.VideoCapture(video_path)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_name = f"{video_name[:-4]}_{frames[0]}_{frames[-1]}.mp4"
    output_path = os.path.join(output_path, "tackle_clips", clip_name)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        if not ret:
            break
        out.write(img)
    out.release()

    # Conver the video to h264 format using moviepy
    clip = moviepy.VideoFileClip(output_path)
    clip.write_videofile(output_path[:-4] + "_h264.mp4", codec="libx264", audio_codec="aac")
    # Delete the original video
    os.remove(output_path)

def make_directories(output_dir, video_name):
    """
    This function will create the directories for the output files of the pipeline. The function will only create
    the directories if they do not already exist.

    Inputs:
        output_dir: The path to the output directory for all the subdirectories
        video_name: The name of the video file
    """
    if not os.path.exists(os.path.join(output_dir, "action_recogniser")):
        os.makedirs(os.path.join(output_dir, "action_recogniser"))
    if not os.path.exists(os.path.join(output_dir, "tackle_clips")):
        os.makedirs(os.path.join(output_dir, "tackle_clips"))
    if not os.path.exists(os.path.join(output_dir, "tackle_frames")):
        os.makedirs(os.path.join(output_dir, "tackle_frames"))
    if not os.path.exists(os.path.join(output_dir, "tackle_frames", video_name[:-4])):
        os.makedirs(os.path.join(output_dir, "tackle_frames", video_name[:-4]))
    if not os.path.exists(os.path.join(output_dir, "tackle_location")):
        os.makedirs(os.path.join(output_dir, "tackle_location"))
    if not os.path.exists(os.path.join(output_dir, "tackle_location", video_name[:-4])):
        os.makedirs(os.path.join(output_dir, "tackle_location", video_name[:-4]))
    if not os.path.exists(os.path.join(output_dir, "tackle_crops")):
        os.makedirs(os.path.join(output_dir, "tackle_crops"))
    if not os.path.exists(os.path.join(output_dir, "tackle_crops", video_name[:-4])):
        os.makedirs(os.path.join(output_dir, "tackle_crops", video_name[:-4]))
    if not os.path.exists(os.path.join(output_dir, "pose_estimation")):
        os.makedirs(os.path.join(output_dir, "pose_estimation"))
    if not os.path.exists(os.path.join(output_dir, "pose_estimation", video_name[:-4])):
        os.makedirs(os.path.join(output_dir, "pose_estimation", video_name[:-4]))
    if not os.path.exists(os.path.join(output_dir, "pose_classification")):
        os.makedirs(os.path.join(output_dir, "pose_classification"))
    if not os.path.exists(os.path.join(output_dir, "pose_classification", video_name[:-4])):
        os.makedirs(os.path.join(output_dir, "pose_classification", video_name[:-4]))

def tackle_location_prediction(yolo_model, frames, tackle_idx, video_name, video_path, output_dir, conf_threshold=0.5):
    """
    This function takes in a list of frames that are suspected to contain a tackle and uses a custom YOLOv8 model
    to predict the bounding box of the tackle. The function will save the bounding box for each frame to the results 
    whilst writing the annotated frames to the output directory. The function will also save the original frames to
    be used in later stages of the pipeline. If multiple bounding boxes are detected in a frame the function will
    select the bounding box with the highest confidence. The function will return the results of the bounding box
    predictions.

    Inputs:
        yolo_model: The custom YOLOv8 model used to predict the bounding box of the tackle
        frames: The list of frames that contain a tackle
        tackle_idx: The index of the tackle
        video_name: The name of the video file
        video_path: The path to the video file
        output_dir: The path to the output directory
        conf_threshold: The confidence threshold for the bounding box predictions
    
    Outputs:
        Original frames saved to the output directory
        Annotated frames saved to the output directory
        all_results: The results of the bounding box predictions across all the frames
    """
    cap = cv2.VideoCapture(video_path)
    all_results = {}
    is_empty = True
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        if not ret:
            break
        results = yolo_model.predict(img, save=False, show=False, conf=conf_threshold, imgsz=640)
        # Save the results
        cv2.imwrite(
            os.path.join(output_dir, "tackle_frames", video_name[:-4], f"tackle_{tackle_idx}", f"frame_{frame}.jpg"),
            img)
        annotated_frame = results[0].plot()
        cv2.imwrite(
            os.path.join(output_dir, "tackle_location", video_name[:-4], f"tackle_{tackle_idx}", f"frame_{frame}.jpg"),
            annotated_frame
        )
        best_confidence = 0
        best_box = None
        for r in results:
            # Check if there are any bounding boxes
            if len(r.boxes) > 0:
                if r.boxes[0].conf > best_confidence:
                    best_confidence = r.boxes[0].conf
                    best_box = r.boxes[0].xyxy
        if best_box is not None:
            all_results[frame] = best_box
            is_empty = False
        else:
            all_results[frame] = []
    return all_results, is_empty

def crop_to_tackle_location(localisation_data, tackle_idx, video_name, video_path, output_dir, padding=0.25):
    """
    This function will crop the frame of the video to the bounding box of the tackle. The function will save the cropped
    image to the output directory. The function will also add padding to the bounding box to ensure that the tackle is
    fully captured in the cropped image.

    Inputs:
        localisation_data: The bounding box of the tackle across all the frames
        tackle_idx: The index of the tackle
        video_name: The name of the video file
        video_path: The path to the video file
        output_dir: The path to the output directory
        padding: The amount of padding to add to the bounding box as a percentage

    Outputs:
        Cropped images saved to the output directory
    """
    cap = cv2.VideoCapture(video_path)
    for frame in localisation_data:
        # Check if the frame contains a bounding box
        if len(localisation_data[frame]) == 0:
            continue
        # Get the bounding box of the tackle
        bounding_box = localisation_data[frame][0].to("cpu").numpy()
        x1, y1, x2, y2 = round(bounding_box[0]), round(bounding_box[1]), round(bounding_box[2]), round(bounding_box[3])
        # Padding is hyperparameter which determines how much padding we want to add to the bounding box
        padding_x = int((x2 - x1) * padding)
        padding_y = int((y2 - y1) * padding)
        # Get the image that we want to crop
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        if not ret:
            break
        # Prevent the padding from going outside of the image
        new_x1 = max(0, int(x1 - padding_x))
        new_x2 = min(img.shape[1], int(x2 + padding_x))
        new_y1 = max(0, int(y1 - padding_y))
        new_y2 = min(img.shape[0], int(y2 + padding_y))
        # Crop the image
        cropped_img = img[new_y1:new_y2, new_x1:new_x2]
        # Save the cropped image
        cv2.imwrite(
            os.path.join(output_dir, "tackle_crops", video_name[:-4], f"tackle_{tackle_idx}", f"frame_{frame}.jpg"),
            cropped_img
        )

def pose_estimation(pose_estimation_model, frame, tackle_idx, tackle_location, video_name, output_dir):
    """
    This function will use the pose estimation model to estimate the keypoints of the players in the tackle. The 
    function will only output the pose estimations for the players that have their hip or shoulder keypoints within
    the bounding box of the tackle. The function will save the annotated frames to the output directory.

    Inputs:
        pose_estimation_model: The pose estimation model used to estimate the keypoints of the players
        frame: The frame number of the video
        tackle_idx: The index of the tackle
        tackle_location: The bounding box of the tackle
        video_name: The name of the video file
        output_dir: The path to the output directory
    
    Outputs:
        Annotated frames saved to the output directory
        players_in_tackle: The keypoints of the players in the tackle
        players_in_tackle_normalised: The normalised keypoints of the players in the tackle
        players_in_tackle_confidence: The confidence of the keypoints of the players in the tackle
    """
    frame_path = os.path.join(
        output_dir,
        "tackle_frames",
        video_name[:-4],
        f"tackle_{tackle_idx}",
        f"frame_{frame}.jpg"
    )
    results = pose_estimation_model.predict(frame_path, save=False, show=False, imgsz=1280)
    players_in_tackle = []
    players_in_tackle_normalised = []
    players_in_tackle_confidence = []
    annotated_frame = results[0].plot()
    cv2.imwrite(
        os.path.join(output_dir, "pose_estimation", video_name[:-4], f"tackle_{tackle_idx}", f"frame_{frame}.jpg"),
        annotated_frame
    )
    bounding_box = tackle_location[frame][0].to("cpu").numpy()
    x1, y1, x2, y2 = round(bounding_box[0]), round(bounding_box[1]), round(bounding_box[2]), round(bounding_box[3])
    for r in results:
        # We want to check to see if the results are empty
        if r.keypoints[0].xy.to("cpu").shape == (1, 0, 2):
            continue
        x1, y1, x2, y2 = round(bounding_box[0]), round(bounding_box[1]), round(bounding_box[2]), round(bounding_box[3])
        # We want to output the pose estimations have their hip or shoulder keypoints in the bounding box of the tackle
        players_in_tackle = []
        players_in_tackle_normalised = []
        players_in_tackle_confidence = []
        for player_keypoint in r.keypoints:
            points = player_keypoint.xy.to("cpu").numpy()[0]
            points_normalised = player_keypoint.xyn.to("cpu").numpy()[0]
            confidence = player_keypoint.conf.to("cpu").numpy()
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
                    or_gate = True
            if or_gate:
                players_in_tackle.append(points)
                players_in_tackle_normalised.append(points_normalised)
                players_in_tackle_confidence.append(confidence)
    print(players_in_tackle)
    return players_in_tackle, players_in_tackle_normalised, players_in_tackle_confidence

def pipeline(
        video_path,
        action_model_path,
        yolo_model_path,
        pose_estimation_model_path,
        pose_classification_model_path,
        output_dir,
        device,
        number_of_clip_frames=16,
        temporal_full_version=True,
        temporal_conf_threshold=0.5,
        group_size=10,
        group_required=15,
        before_buffer=10,
        after_buffer=10,
        spatial_conf_threshold=0.5
):
    """
    This function is the main pipeline for the project. It passes all of the data through the pipeline to detect the
    tackles in the video and classify the legality of the tackles based on the domestic law variation. The function
    will output the results of the pipeline to a results.csv file which will be saved in the output directory. It will
    also output annotated frames and videos throughout the pipeline to the output directory.

    Inputs:

    Outputs:
        The results of the pipeline saved to a results.csv file in the output directory
        Annotated additional files saved to the output directory
    """
    # Load the models
    action_model = torch.load(action_model_path, map_location=device).to(device)
    # YOLO automatically puts the model onto the correct device
    yolo_model = YOLO(yolo_model_path)
    pose_estimation_model = YOLO(pose_estimation_model_path)
    pose_classification_model = torch.load(pose_classification_model_path, map_location=device).to(device)
    # All videos are of the name Tackle_XXX.mp4 so we can extract the name of the video from the path
    # If the tackle number is less than 200 it is a low tackle clip otherwise it is a high tackle clip
    # Class 1 - Low Tackle
    # Class 2 - High Tackle
    video_name = f"{video_path.split(os.sep)[-1][:-4]}.mp4"
    tackle_number = video_name.split("_")[1][:-4]
    tackle_number = 0 if tackle_number == "" else int(tackle_number)
    clip_ground_truth = 2 if tackle_number >= 200 else 1
    ground_truth = clip_ground_truth
    results_csv_header = [
        "Video Name",
        "Tackle Number",
        "Tackle Frames",
        "Tackles Located",
        "People in Tackle",
        "Tackle Classification",
        "Ground Truth",
        "Ground Truth Tackle Start Frames",
        "Ground Truth Tackle End Frames"
    ]
    with open(os.path.join(output_dir, "tackles.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["video_name"] == video_name:
                ground_truth_start_frame = int(row["start_frame"])
                ground_truth_end_frame = int(row["end_frame"])
                break

    # Make the directories for the outputs
    make_directories(output_dir, video_name)

    # Determine if any of the frames are likely to contain a tackle using our action recogniser model
    frames  = action_prediction(
        action_model,
        video_path,
        output_dir,
        is_full_version=temporal_full_version,
        confidence_threshold=temporal_conf_threshold,
        frame_length=number_of_clip_frames,
        group_size=group_size,
        group_required=group_required
    )
    if len(frames) == 0:
        print("No tackles detected")
        with open(os.path.join(output_dir, "results.csv"), "a", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_csv_header)
            writer.writerow({"Video Name": video_name,
                            "Tackle Number": 1, 
                            "Tackle Frames": -1, 
                            "Tackles Located": -1, 
                            "People in Tackle": -1, 
                            "Tackle Classification": -1,
                            "Ground Truth": ground_truth, 
                            "Ground Truth Tackle Start Frames": ground_truth_start_frame, 
                            "Ground Truth Tackle End Frames": ground_truth_end_frame})
        return

    # Get the maximum number of frames in the video
    cap = cv2.VideoCapture(video_path)
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # We now need to pad out the selected frame by frames either side to ensure that we have enough frames
    # to accurately localise the tackle
    tackle_start_buffered = max(0, ground_truth_start_frame - before_buffer)
    tackle_end_buffered = min(max_frames, ground_truth_end_frame + after_buffer)
    tackle_clips = create_buffered_clips(frames, before_buffer, after_buffer, max_frames)

    # Documenting the frames that contain tackles by writing them to their own video file using OpenCV
    for clip in tackle_clips:
        # We want to write the frames to a new video file
        write_tackle_clips(clip, video_name, video_path, output_dir)

    # If we have detected a tackle we will use a custom YOLOv8 model to detect the bounding box of the tackle
    tackle_localisation = {}
    for idx, tackle in enumerate(tackle_clips):
        ground_truth = clip_ground_truth
        # Create the directories for the specific tackle idx
        if not os.path.exists(os.path.join(output_dir, "tackle_frames", video_name[:-4], f"tackle_{idx}")):
            os.makedirs(os.path.join(output_dir, "tackle_frames", video_name[:-4], f"tackle_{idx}"))
        if not os.path.exists(os.path.join(output_dir, "tackle_location", video_name[:-4], f"tackle_{idx}")):
            os.makedirs(os.path.join(output_dir, "tackle_location", video_name[:-4], f"tackle_{idx}"))
        if not os.path.exists(os.path.join(output_dir, "tackle_crops", video_name[:-4], f"tackle_{idx}")):
            os.makedirs(os.path.join(output_dir, "tackle_crops", video_name[:-4], f"tackle_{idx}"))
        # Check if the tackle and the buffered tackle overlap
        overlap = range(max(tackle_start_buffered, tackle[0]), min(tackle_end_buffered, tackle[-1]))
        print("Overlap: ", overlap)
        if len(overlap) == 0:
            print("No overlap")
            ground_truth = -1
        results, prediction_empty = tackle_location_prediction(
            yolo_model,
            tackle,
            idx,
            video_name,
            video_path,
            output_dir,
            conf_threshold=spatial_conf_threshold
        )
        if not prediction_empty:
            tackle_localisation[idx] = results
        else:
            with open(os.path.join(output_dir, "results.csv"), "a", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results_csv_header)
                writer.writerow({"Video Name": video_name,
                                "Tackle Number": idx, 
                                "Tackle Frames": tackle[0], 
                                "Tackles Located": 0, 
                                "People in Tackle": -1, 
                                "Tackle Classification": -1,
                                "Ground Truth": ground_truth, 
                                "Ground Truth Tackle Start Frames": ground_truth_start_frame,
                                "Ground Truth Tackle End Frames": ground_truth_end_frame})

    # Now check if tackle_localisation is empty
    if len(tackle_localisation) == 0:
        print("No tackles detected")
        return
    for data in tackle_localisation.items():
        tackle = data[0]
        ground_truth = clip_ground_truth
        crop_to_tackle_location(tackle_localisation[tackle], tackle, video_name, video_path, output_dir)

        # Make the specific tackle directories for the pose estimation and classification per video
        if not os.path.exists(os.path.join(output_dir, "pose_estimation", video_name[:-4], f"tackle_{tackle}")):
            os.makedirs(os.path.join(output_dir, "pose_estimation", video_name[:-4], f"tackle_{tackle}"))
        if not os.path.exists(os.path.join(output_dir, "pose_classification", video_name[:-4], f"tackle_{tackle}")):
            os.makedirs(os.path.join(output_dir, "pose_classification", video_name[:-4], f"tackle_{tackle}"))

        tackle_prediction = []
        # Apply pose estimation on the original tackle frames
        average_number_of_players = 0

        first_frame = list(tackle_localisation[tackle].keys())[0]
        last_frame = list(tackle_localisation[tackle].keys())[-1]
        overlap = range(max(tackle_start_buffered, first_frame), min(tackle_end_buffered, last_frame))
        print("Overlap: ", overlap)
        if len(overlap) == 0:
            print("No overlap")
            ground_truth = -1

        for frame in tackle_localisation[tackle]:
            # Check if the frame detected a tackle
            if len(tackle_localisation[tackle][frame]) == 0:
                continue
            players_in_tackle, players_in_tackle_normalised, players_in_tackle_confidence = pose_estimation(
                pose_estimation_model,
                frame,
                tackle,
                tackle_localisation[tackle],
                video_name,
                output_dir
            )
            average_number_of_players += len(players_in_tackle)
            if len(players_in_tackle) < 2:
                print("Not enough players in the tackle")
                continue
            # We now want to loop through the pairs of players in the tackle and classify the pose of the players
            for play1, player1 in enumerate(players_in_tackle):
                for play2, player2 in enumerate(players_in_tackle, start=play1 + 1):
                    player1_normalised = players_in_tackle_normalised[play1]
                    player2_normalised = players_in_tackle_normalised[play2]
                    player1_confidence = players_in_tackle_confidence[play1]
                    player2_confidence = players_in_tackle_confidence[play2]

                    player2_head_x, player2_head_y, player1_distances = compute_player_distances(player2_normalised, player1_normalised)
                    player1_head_x, player1_head_y, player2_distances = compute_player_distances(player1_normalised, player2_normalised)
                    # Check if the distance arrays are empty
                    if player1_distances.size == 0 or player2_distances.size == 0:
                        # No head estimation could be found which means that it is not possible to classify the tackles
                        print("No head estimation could be found. Not able to make a classification!")
                        continue

                    img = cv2.imread(os.path.join(
                                        output_dir,
                                        "tackle_frames",
                                        video_name[:-4],
                                        f"tackle_{tackle}", f"frame_{frame}.jpg"
                                    ))
                    # Convert the normalised keypoints to the original image size
                    dh, dw, _ = img.shape
                    player1_head_x = int(player1_head_x * dw)
                    player1_head_y = int(player1_head_y * dh)
                    player2_head_x = int(player2_head_x * dw)
                    player2_head_y = int(player2_head_y * dh)

                    player1 = np.insert(player1, 2, player1_confidence, axis=1)
                    player2 = np.insert(player2, 2, player2_confidence, axis=1)
                    annotator = Annotator(img, line_width=2, font_size=1.0)
                    # Draw the annotations
                    annotator.kpts(player1)
                    annotator.kpts(player2)
                    out_img = annotator.result()

                    # Plot the estimated head location of the players
                    cv2.circle(out_img, (int(player1_head_x), int(player1_head_y)), 5, (0, 0, 255), -1)
                    cv2.circle(out_img, (int(player2_head_x), int(player2_head_y)), 5, (0, 0, 255), -1)

                    # We want to classify the pose of the players in the tackle
                    predicted_class = tackle_classification(
                        pose_classification_model,
                        player1_distances,
                        player2_distances,
                        device
                    )
                    tackle_prediction.append(predicted_class.item())
                    # We want to write the classification onto the image
                    # This is in BGR colour format
                    tackle_type = ""
                    colour = (0, 0, 0)
                    if predicted_class == 0:
                        tackle_type = "No Tackle"
                        colour = (255, 0, 0)
                        print("No Tackle")
                    elif predicted_class == 1:
                        tackle_type = "Low Tackle"
                        colour = (0, 255, 0)
                        print("Low Tackle")
                    elif predicted_class == 2:
                        tackle_type = "High Tackle"
                        colour = (0, 0, 255)
                        print("High Tackle")
                    cv2.putText(out_img, tackle_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                    # Save the image
                    cv2.imwrite(
                        os.path.join(
                            output_dir,
                            "pose_classification",
                            video_name[:-4],
                            f"tackle_{tackle}", f"frame_{frame}_player{play1}_player{play2}.jpg"
                        ),
                        out_img
                    )

            # The tackle prediction is the maximum class that was predicted for the players in the tackle
        number_of_located_tackles = len([i for i in tackle_localisation[tackle].values() if i != []])
        average_number_of_players /= number_of_located_tackles
        print(f"Video Name: {video_name} Tackle Number: {tackle}  Tackle Prediction: {tackle_prediction}")
        if len(tackle_prediction) == 0:
            print("No tackle prediction")
            with open(os.path.join(output_dir, "results.csv"), "a", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results_csv_header)
                writer.writerow({"Video Name": video_name,
                                "Tackle Number": tackle, 
                                "Tackle Frames": list(tackle_localisation[tackle].keys())[0],
                                "Tackles Located": number_of_located_tackles,
                                "People in Tackle": average_number_of_players,
                                "Tackle Classification": -1,
                                "Ground Truth": ground_truth, 
                                "Ground Truth Tackle Start Frames": ground_truth_start_frame,
                                "Ground Truth Tackle End Frames": ground_truth_end_frame})
            continue
        # The predicted class is the class that occurs the most in the list
        # With class 0 only being predicted if at least 80% of the predictions are class 0
        class_0_number = 0
        class_1_number = 0
        class_2_number = 0
        total_number = 0
        for i in tackle_prediction:
            if i == 0:
                class_0_number += 1
            elif i == 1:
                class_1_number += 1
            elif i == 2:
                class_2_number += 1
        total_number = class_0_number + class_1_number + class_2_number
        if (class_1_number == 0 and class_2_number == 0) or class_0_number > 0.8 * total_number:
            tackle_prediction = 0
        elif class_1_number > class_2_number:
            tackle_prediction = 1
        else:
            tackle_prediction = 2
        with open(os.path.join(output_dir, "results.csv"), "a", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_csv_header)
            writer.writerow({"Video Name": video_name,
                            "Tackle Number": tackle, 
                            "Tackle Frames": list(tackle_localisation[tackle].keys())[0],
                            "Tackles Located": number_of_located_tackles,
                            "People in Tackle": average_number_of_players,
                            "Tackle Classification": tackle_prediction,
                            "Ground Truth": ground_truth, 
                            "Ground Truth Tackle Start Frames": ground_truth_start_frame,
                            "Ground Truth Tackle End Frames": ground_truth_end_frame})


def main():
    """
    This is the main function which will parse the system arguments and run the pipeline on all video files in within
    the input directory. The function will output the results of the pipeline to a results.csv file which will be saved
    in the output directory. It will also output annotated frames and videos throughout the pipeline to the output
    directory. 

    Arguments:
        --dir_path: The path to the directory containing the video files
        --output_dir: The path to the output directory for the results
        --temporal_locator: The path to the temporal locator model
        --spatial_locator: The path to the spatial locator model
        --pose_estimator: The path to the pose estimator model
        --pose_classifier: The path to the pose classifier model
        --number_of_clip_frames: The number of frames in each clip
        --temporal_full_version: Whether to use the full version of the temporal locator
        --temporal_conf_threshold: The confidence threshold for the temporal locator
        --group_size: The size of the group for the temporal locator
        --group_required: The number of groups required for the temporal locator
        --before_buffer: The number of frames before the selected frame
        --after_buffer: The number of frames after the selected frame
        --spatial_conf_threshold: The confidence threshold for the spatial locator
    """
    parser = argparse.ArgumentParser(description="Pipeline for detecting tackles in rugby videos")
    parser.add_argument("--dir_path", type=str, help="The path to the directory containing the video files")
    # Default: /dcs/large/u2102661/CS310/datasets/activity_recogniser/original_clips
    parser.add_argument("--output_dir", type=str, help="The path to the output directory for the results")
    # Default: /dcs/large/u2102661/CS310/model_evaluation/pipeline_original_clips
    parser.add_argument("--temporal_locator", type=str, help="The path to the temporal locator model")
    # Default: /dcs/large/u2102661/CS310/models/activity_recogniser/r3d_18/5_frames/best.pt
    parser.add_argument("--spatial_locator", type=str, help="The path to the spatial locator model")
    # Default: /dcs/large/u2102661/CS310/models/tackle_location/train5/weights/best.pt
    parser.add_argument("--pose_estimator", type=str, help="The path to the pose estimator model")
    # Default: yolov8x-pose-p6.pt
    parser.add_argument("--pose_classifier", type=str, help="The path to the pose classifier model")
    # Default: /dcs/large/u2102661/CS310/models/pose_estimation/run_new_data_12/best.pt
    parser.add_argument("--number_of_clip_frames", type=int, help="The number of frames in each clip")
    # Default: 16
    parser.add_argument("--temporal_full_version", type=bool, help="Whether to use the full version of the temporal locator")
    # Default: True
    parser.add_argument("--temporal_conf_threshold", type=float, help="The confidence threshold for the temporal locator")
    # Default: 0.5
    parser.add_argument("--group_size", type=int, help="The size of the group for the temporal locator")
    # Default: 10
    parser.add_argument("--group_required", type=int, help="The number of groups required for the temporal locator")
    # Default: 15
    parser.add_argument("--before_buffer", type=int, help="The number of frames before the selected frame")
    # Default: 10
    parser.add_argument("--after_buffer", type=int, help="The number of frames after the selected frame")
    # Default: 10
    parser.add_argument("--spatial_conf_threshold", type=float, help="The confidence threshold for the spatial locator")
    # Default: 0.5

    args = parser.parse_args()
    args.dir_path = "/dcs/large/u2102661/CS310/datasets/full_clip"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_csv_header = [
        "Video Name",
        "Tackle Number",
        "Tackle Frames",
        "Tackles Located",
        "People in Tackle",
        "Tackle Classification",
        "Ground Truth",
        "Ground Truth Tackle Start Frames",
        "Ground Truth Tackle End Frames"
    ]
    with open(os.path.join(args.output_dir, "results.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(results_csv_header)
    if not os.path.exists(os.path.join(args.dir_path, "..", "done")):
        os.makedirs(os.path.join(args.dir_path, "..", "done"))
    for file in os.listdir(args.dir_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(args.dir_path, file)
            print(f"===================={file}====================")
            pipeline(
                video_path,
                args.temporal_locator,
                args.spatial_locator,
                args.pose_estimator,
                args.pose_classifier,
                args.output_dir,
                device,
                number_of_clip_frames=args.number_of_clip_frames,
                temporal_full_version=args.temporal_full_version,
                temporal_conf_threshold=args.temporal_conf_threshold,
                group_size=args.group_size,
                group_required=args.group_required,
                before_buffer=args.before_buffer,
                after_buffer=args.after_buffer,
                spatial_conf_threshold=args.spatial_conf_threshold
            )
            # Moves the file to the done folder incase the job crashes
            os.rename(video_path, os.path.join(os.path.join(args.dir_path, "..", "done", file)))

if __name__ == "__main__":
    main()
