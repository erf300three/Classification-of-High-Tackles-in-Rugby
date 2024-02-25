import os 
import sys
import csv
import cv2
import copy
import math
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.io import read_video, VideoReader
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, v2

class ActionRecogniserDataset(Dataset):
    def __init__(self, annotation_file, video_dir, is_test=False, transform=None, target_transform=None):
        self.annotation_file = pd.read_csv(annotation_file)
        self.video_dir = video_dir
        self.is_test = is_test
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.annotation_file.iloc[idx, 0])
        video_name = self.annotation_file.iloc[idx, 0]
        video, audio, info = read_video(video_path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)
        video = video.to(dtype=torch.float32)
        label = self.annotation_file.iloc[idx, 3]
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        video = video.permute(1, 0, 2, 3)
        return video, label, video_name

class ActionRecogniserDatasetFullVideoClips(Dataset):
    def __init__(self, annotation_file, video_dir, transform=None, target_transform=None):
        self.annotation_file = pd.read_csv(annotation_file)
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotation_file)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.annotation_file.iloc[idx, 0])
        video_name = self.annotation_file.iloc[idx, 0]
        video, audio, info = read_video(video_path, pts_unit='sec')
        video = video.to(device)
        video = video.permute(0, 3, 1, 2)
        # video = video.to(dtype=torch.float32)
        print(video.shape)
        label = (self.annotation_file.iloc[idx, 1], self.annotation_file.iloc[idx, 2])
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        video = video.permute(1, 0, 2, 3)
        # We are going to unfold the video into 16 frame clips
        # video = video.unfold(1, 16, 16).permute(0, 2, 1, 3, 4)
        return video, label, video_name

# There is a particular error with the r3d_18 model where a flatten layer is missing when you call the children() method
# This flatten method needs to go before the final linear layer of the model. To fix this we will manually insert the flatten layer
# into the model
# GitHub Issue: https://github.com/pytorch/vision/issues/4083

class ActionRecogniserModel(nn.Module):
    def __init__(self, parent_model=None, parent_model_name=""):
        super(ActionRecogniserModel, self).__init__()
        # We want to keep the final layer of the parent model
        
        # If we want to use a different parent model
        if parent_model == None:
            self.parent_model = nn.Sequential(*list(models.video.r3d_18(pretrained=True).children())[:-1], nn.Flatten(1), list(models.video.r3d_18(pretrained=True).children())[-1])
        elif parent_model_name == "r3d_18":
            self.parent_model = nn.Sequential(*list(models.video.r3d_18(pretrained=True).children())[:-1], nn.Flatten(1), list(models.video.r3d_18(pretrained=True).children())[-1])
        else:
            self.parent_model = nn.Sequential(*list(parent_model.children()))
        self.batchNorm1 = nn.BatchNorm1d(400)
        self.fc1 = nn.Linear(400, 200)
        self.batchNorm2 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.batchNorm3 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.parent_model(x)
        x.squeeze_()
        x = self.batchNorm1(x)
        x = self.relu(self.fc1(x))
        x = self.batchNorm2(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batchNorm3(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

transform_main = Compose([
    (lambda x: x / 255),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    RandomRotation(degrees=15),
    RandomHorizontalFlip(p=0.5) #, 
    # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])

transform1 = Compose([
    (lambda x: x / 255),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

# We have added normalisation to the model so we need to normalise the data before we pass it to the model
# This is with values that are outlined in the documentation for the kinetics dataset found here
# https://github.com/pytorch/vision/blob/master/references/video_classification/train.py#L115
train_data = ActionRecogniserDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/activity_recogniser/train/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/activity_recogniser/train",
    is_test=False, 
    transform=transform_main
)
test_data = ActionRecogniserDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/activity_recogniser/test/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/activity_recogniser/test",
    is_test=True, 
    transform=transform1
)
validate_data = ActionRecogniserDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/activity_recogniser/validation/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/activity_recogniser/validation",
    is_test=True, 
    transform=transform1
)

# =============================================================================
# Calculate the class weights for the loss function
# =============================================================================
# Class 0: Non-tackle
# Class 1: Tackle

# # Location    Number of tackle clips    Number of non-tackle clips    Total number of clips
# # All dirs:   570                       11337                         11907
# # Train:      442                       9027                          9469
# # Test:       72                        1143                          1215
# # Validation: 56                        1167                          1223

# weight_class_i = #total_clips / (#classes * #clips_in_class_i)

# All dirs:   11907 / (2 * 570) = 10.44 , 11907 / (2 * 11337) = 0.525
# Train:      9469 / (2 * 442) = 10.71 , 9469 / (2 * 9027) = 0.524
# Test:       1215 / (2 * 72) = 8.44   , 1215 / (2 * 1143) = 0.531
# Validation: 1223 / (2 * 56) = 10.92 , 1223 / (2 * 1167) = 0.524

# We are going to use the class weights for the training data 
class_weights = [0.524, 10.71]
class_weights = [0.262, 21.42]

# We are going to follow the same intuition that pytorch uses for the BCELoss function
# Since log(0) = -inf as lim x->0 log(x) = -inf and then we could be multiplying by 0 we can be 
# in a situation where we get a loss value of nan and a gradient of lim x-> 0 ( d/dx(log(x)) = inf
# which would make the backward method non linear. 
# 
# To avoid this problem we are going to do what clamp the log value to be greater than -100 
def loss_function(output, target):
    # print("output: ", output)
    # print("target: ", target)

    temp1 = torch.clamp(torch.log(output), min=-100)
    temp2 = torch.clamp(torch.log(1 - output), min=-100)

    all_loss = -(10.71 * (target * temp1) + 0.524 * ((1 - target) * temp2))
    # print("(clamped) log(output) : ", temp1)
    # print("(clamped) log(1 - output): ", temp2)
    # print("target * torch.log(output): ", target * temp1)
    # print("(1 - target) * torch.log(1 - output): ", (1 - target) * temp2)
    # print("all_loss", all_loss)
    loss = torch.mean(all_loss)
    # print("loss: ", loss)
    return loss

def train(model, train_loader, loss_fun, optimizer, batch_size=4):
    model.train()
    size = len(train_loader.dataset)
    computed_so_far = 0
    mean_loss = []
    print("Training...")
    for batch_idx, (data, target, video_name) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        computed_so_far += len(data)
        target = target.unsqueeze(1).to(dtype=torch.float32)
        output = model(data)
        loss = loss_fun(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mean_loss.append(loss.item())
        loss = loss.item()
        print(f"loss: {loss:>7f} batch size: {len(data)}  [{computed_so_far:>5d}/{size:>5d}]")
    mean_loss = np.mean(mean_loss)
    print(f"Mean loss: {mean_loss:>7f}")
    return mean_loss

def validate(model, test_loader, batch_size=4):
    model.eval()
    print("Validating...")
    size = len(test_loader.dataset) 
    computed_so_far = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0  
    # For each batch in the test set we need to calculate the TP, TN, FP, FN
    with torch.no_grad():
        for batch_idx, (data, target, video_name) in enumerate(test_loader):
            computed_so_far += len(data)
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1).to(dtype=torch.float32)
            
            output = model(data)

            batch_TP = 0
            batch_TN = 0
            batch_FP = 0
            batch_FN = 0

            # Calculate the TP, TN, FP, FN for this batch
            for i in range(len(output)):
                if output[i] >= 0.5 and target[i] == 1:
                    true_positives += 1
                    batch_TP += 1
                elif output[i] >= 0.5 and target[i] == 0:
                    false_positives += 1
                    batch_FP += 1
                elif output[i] < 0.5 and target[i] == 1:
                    false_negatives += 1
                    batch_FN += 1
                elif output[i] < 0.5 and target[i] == 0:
                    true_negatives += 1
                    batch_TN += 1

            # We could have the case where there are no positive clips in the batch
            if batch_TP + batch_FN== 0:
                batch_f1_score = 0
                print(f"batch f1 score: N/A  batch size: {len(data)}  [{computed_so_far:>5d}/{size:>5d}]")
            elif batch_TP + batch_FP == 0:
                batch_f1_score = 0
                print(f"batch f1 score: N/A  batch size: {len(data)}  [{computed_so_far:>5d}/{size:>5d}]")
            else:
                recall = batch_TP / (batch_TP + batch_FN)
                precision = batch_TP / (batch_TP + batch_FP)
                batch_f1_score = 0
                if recall + precision != 0:
                    batch_f1_score = 2 * (precision * recall / (precision + recall))
                print(f"recall: {recall:>7f} precision: {precision:>7f}")
                print(f"batch f1 score: {batch_f1_score:>7f} batch size: {len(data)}  [{computed_so_far:>5d}/{size:>5d}]")  
        
    print(f"true positives: {true_positives}")
    print(f"true negatives: {true_negatives}")
    print(f"false positives: {false_positives}")
    print(f"false negatives: {false_negatives}")

    if true_positives + false_negatives == 0:
        full_f1_score = 0 
        full_f2_score = 0
    elif true_positives + false_positives == 0:
        full_f1_score = 0
        full_f2_score = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        full_f1_score = 0
        full_f2_score = 0
        if precision + recall != 0:
            full_f1_score = 2 * (precision * recall / (precision + recall))
            full_f2_score = 5 * (precision * recall / (4 * precision + recall))


    print(f"full f1 score: {full_f1_score:>7f}")
    print(f"full f2 score: {full_f2_score:>7f}")

    return full_f1_score, full_f2_score

def resize_video(input_tensor, size):
    return torch.stack([v2.Resize(size)(input_tensor[i, :, :, :, :]) for i in range(input_tensor.shape[0])], dim=0) 

def evaluate(model, test_loader, batch_stize=16):
    model.eval()
    print("Evaluating...")
    size = len(test_loader)
    number_completed = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    with open("/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/evaluation.csv", 'w', newline='') as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["video_name", "output", "target"])
        csv_file.writeheader()
        with torch.no_grad():
            for batch_idx, (data, target, video_name) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                target = target.unsqueeze(1).to(dtype=torch.float32)
                output = model(data)
                for i in range(len(output)):
                    if output[i] >= 0.5 and target[i] == 1:
                        true_positives += 1
                    elif output[i] >= 0.5 and target[i] == 0:
                        false_positives += 1
                    elif output[i] < 0.5 and target[i] == 1:
                        false_negatives += 1
                    elif output[i] < 0.5 and target[i] == 0:
                        true_negatives += 1
                number_completed += len(data)
                for i in range(len(output)):
                    csv_file.writerow({"video_name": video_name[i], "output": output[i].item(), "target": target[i].item()})
                print(f"batch size: {len(data)}  [{number_completed:>5d}/{size:>5d}]")
    print(f"true positives: {true_positives}")
    print(f"true negatives: {true_negatives}")
    print(f"false positives: {false_positives}")
    print(f"false negatives: {false_negatives}")
    f1_score = 0
    f2_score = 0
    recall = 0
    precision = 0
    if true_positives + false_negatives != 0:
        recall = true_positives / (true_positives + false_negatives)
    if true_positives + false_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    if precision + recall != 0:
        f1_score = 2 * (precision * recall / (precision + recall))
        f2_score = 5 * (precision * recall / (4 * precision + recall))
    with open("/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/results.csv", 'w', newline='') as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["true_positives", "true_negatives", "false_positives", "false_negatives", "recall", "precision", "f1_score", "f2_score"])
        csv_file.writeheader()
        csv_file.writerow({
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "recall": recall, 
            "precision": precision, 
            "f1_score": f1_score, 
            "f2_score": f2_score
        })
    
def full_video_clips_evaluate(model, dir_to_test):
    model.eval()
    print("Testing full video clips...")

    full_length_clip_transform = Compose([
        (lambda x: x / 255),
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989], inplace=True)
    ])

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
    
    tackle_data = transform_csv_to_object(dir_to_test)

    with open("/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/full_video_clips.csv", 'w', newline='') as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["video_name", "frame_number", "output"])
        csv_file.writeheader()
        for video_name in tackle_data:
            video_path = os.path.join(dir_to_test, video_name)
            
            # Read the video and split it into 16 frame clips 
            video, audio, info = read_video(video_path, pts_unit='sec')
            video = video.to(device)
            video = video.permute(0, 3, 1, 2)
            print(video.shape)
            
            # unfold the video into 16 frame clips that overlap 
            video = video.unfold(0, 16, 1).permute(1, 0, 4, 2, 3)

            print("video shape: ", video.shape)
            concat_videos = None
            # Implementing our own batches of 16
        
            for i in range(0, video.shape[1], 16):
                if i + 16 > video.shape[1]:
                    concat_videos = torch.stack([video[:, i + j, :, :, :] for j in range(0, video.shape[1] - i)], dim = 0)
                    print(concat_videos.shape)
                else:
                    concat_videos = torch.stack([video[:, i + j , :, :, :] for j in range(16)], dim = 0)
                    print(concat_videos.shape)

                # Apply the transform to the video
                float_video = concat_videos.to(dtype=torch.float32)
                float_video = float_video.permute(0, 2, 1, 3, 4)
                resized_video = resize_video(float_video, (224, 224))
                transformed_video = full_length_clip_transform(resized_video)
                transformed_video = transformed_video.permute(0, 2, 1, 3, 4)
                print(transformed_video.shape)

                del float_video
                del resized_video
                del concat_videos
                print(torch.cuda.memory_allocated() / 1024 ** 3)

                with torch.no_grad():
                    output = model(transformed_video)
                    print(output)

                    for j in range(len(output)):
                        csv_file.writerow({"video_name": video_name, "frame_number": i + j, "output": output[j].item()})

                del transformed_video
                del output
                print(torch.cuda.memory_allocated() / 1024 ** 3)

            del video
    print("Done testing full video clips")

# This is the function that will be used in the pipeline to predict the frames of a video that is likely to contain a tackle
# We are again using a sliding window of 16 frames which will be split between 8 frames before and 7 frames after the current frame
def action_prediction(model, video_path):
    print("Predicting frames likely to contain a tackle...")

    # We are going to need to read the video in chunks of 16 frames as a tensor
    frames = []
    # This reader is just used to get the metadata of the video so we can iterate through the frames correctly
    reader = VideoReader(video_path, "video")
    clip = None
    # Our main loop which centres the sliding window
    clips = []
    outputs = []

    for i in range(8, int(reader.get_metadata()["video"]["fps"][0] * reader.get_metadata()["video"]["duration"][0]) - 8):
        clip = None
        print("i: ", i)
        print("start_pts: ", (i-8) * float(1 / reader.get_metadata()["video"]["fps"][0]))
        print("end_pts: ", (i+7) * float(1 / reader.get_metadata()["video"]["fps"][0]))
        
        clip = read_video(video_path, pts_unit='sec', start_pts=(i-8) * float(1 / reader.get_metadata()["video"]["fps"][0]), end_pts=(i+7) * float(1 / reader.get_metadata()["video"]["fps"][0]))
        print("clip shape: ", clip[0].shape)
        # If the length of the clip is greater than 16 then we need to drop the first frames
        if clip[0].shape[0] > 16:
            clip = clip[0][-16:, :, :, :]
        else: 
            clip = clip[0]
        clip = clip.permute(0, 3, 1, 2)
        clip = clip.to(dtype=torch.float32)
        clip = clip.to(device)
        # We are going to need to transform the clip to be in the correct format for the model
        clip = transform1(clip)
        clip = Resize((224, 224))(clip)
        clip = clip.permute(1, 0, 2, 3)
        print("clip shape augmented: ", clip.shape)

        clips.append(clip)
        print("stop condition: ", i % 16)
        if i % 16 == 7 or i == int(reader.get_metadata()["video"]["fps"][0] * reader.get_metadata()["video"]["duration"][0]) - 9:
            # If there is less than 4 frames remain at the end then we just skip them and make no predictions on those frames 
            if len(clips) < 4: 
                clips = []
                continue
            temp = torch.stack(clips, dim=0)
            # Delete all the entries in clip to free up memory
            for clip in clips:
                del clip
            clips = temp
            print(clips.shape)
            with torch.no_grad():
                output = model(clips)
                print(output)
            clips = []
             
            for j in range(len(output)):
                outputs.append(output[j].item())
    
    print("Done predicting frames likely to contain a tackle")

    # Now we need to write the outputs to a csv file 
    with open(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/action_recogniser/{video_path.split('/')[-1][:-4]}.csv", 'w', newline='') as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["frame_number", "output"])
        csv_file.writeheader()
        for i in range(len(outputs)):
            csv_file.writerow({"frame_number": i + 8, "output": outputs[i]})
    

    # We now want to return frames where there are 4 predictions within 16 frames that all are above 0.85
    with open(f"/dcs/large/u2102661/CS310/model_evaluation/pipeline/action_recogniser/{video_path.split('/')[-1][:-4]}.csv", 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        outputs = []
        for row in reader:
            outputs.append((int(row["frame_number"]), float(row["output"])))

    temp = [outputs[i] for i in range(len(outputs)) if outputs[i][1] >= 0.85]    
    # Of these frames we want to keep ones that have another 4 frame within 16 frames of them (the group size will be at least 4)
    def group_numbers(numbers, max_difference=16):
        groups = []
        for number in numbers:
            found_group = False
            for group in groups:
                for member in group:
                    if abs(member - number) <= max_difference:
                        group.append(number)
                        found_group = True
                        break

                    # remove this if-block if a number should be added to multiple groups
                    if found_group:
                        break
            if not found_group:
                groups.append([number])
        return groups    
    groups = group_numbers([i[0] for i in temp])

    # As all of the frames are sequential we can just take the first frame of each group
    return_results = [group[0] for group in groups if len(group) >= 4]
    return return_results

def write_new_films_with_predictions(eval_csv, video_dir):

    def get_data_object(in_path):
        output_file = in_path
        fields = ["video_name", "frame_number", "output"]
        data = {}
        with open(output_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["video_name"] not in data:
                    data[row["video_name"]] = []
                data[row["video_name"]].append((int(row["frame_number"]), float(row["output"])))
        return data

    eval_data = get_data_object(eval_csv)

    print("Writing new videos with predictions...")

    for video_name in eval_data:
        cv2_video = cv2.VideoCapture(os.path.join(video_dir, video_name))
        fps = cv2_video.get(cv2.CAP_PROP_FPS)
        width = int(cv2_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cv2_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = "/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/" + video_name[:-4] + "_output.mp4" 
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        while cv2_video.isOpened():
            ret, frame = cv2_video.read()
            if not ret:
                break
            frame_number = int(cv2_video.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_number <= len(eval_data[video_name]):
                output = eval_data[video_name][(frame_number - 1)][1]
                if output >= 0.5:
                    frame = cv2.putText(frame, "Tackle prediction: " + str(output), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # cv2.imwrite(os.path.join("/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/frames", video_name[:-4] + f"_{frame_number - 1}.jpg"), frame)
                    # print("Done")
                elif output < 0.5:
                    frame = cv2.putText(frame, "Tackle prediction: " + str(output), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.imwrite(os.path.join("/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/frames", video_name[:-4] + f"_{frame_number - 1}.jpg"), frame)
                    # print("Done")
            out.write(frame)

        cv2_video.release()
        out.release()
    cv2.destroyAllWindows()

def save_model(model, path):
    torch.save(model, path)

def main(is_test=False, parent_model_name="r3d_18"):
    # Create the data loaders

    my_model = torch.load("/dcs/large/u2102661/CS310/models/activity_recogniser/r3d_18/run4/best.pt", map_location=device).to(device)
    action_prediction(my_model, "/dcs/large/u2102661/CS310/datasets/initial_set/Bury_v_North_Walsham_Yellow_Card_HIgh_Tackle1.mp4")

    return

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    validate_loader = DataLoader(validate_data, batch_size=16, shuffle=True)

    if is_test:
        # torch.cuda.empty_cache()
        if parent_model_name == "r3d_18":
            my_model = torch.load("/dcs/large/u2102661/CS310/models/activity_recogniser/r3d_18/run4/best.pt")
        my_model = my_model.to(device)
        # evaluate(my_model, test_loader, batch_stize=16)
        print("model loaded")
        # full_video_clips_evaluate(my_model, "/dcs/large/u2102661/CS310/datasets/activity_recogniser/full_length_clips/")
        write_new_films_with_predictions("/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/full_video_clips.csv", "/dcs/large/u2102661/CS310/datasets/activity_recogniser/full_length_clips")
    else:
        # Create the model
        model = models.video.r3d_18(pretrained=True)
        my_model = ActionRecogniserModel(model, "r3d_18").to(device)
        # Freeze the weights of the parent model
        my_model.parent_model.requires_grad_(False)
        best_f1 = -1
        best_f2 = -1
        number_since_best = 0
        best_epoch = 0
        # Print initial weights of last layer
        params = list(my_model.parameters())
        print(params[-2])
        print(params[-1])

        optimizer = torch.optim.Adam(my_model.parameters(), lr = 0.001)

        # This is the training loop
        with open("/dcs/large/u2102661/CS310/models/activity_recogniser/loss.csv", 'w', newline='') as csvfile:
            csv_file = csv.DictWriter(csvfile, fieldnames=["epoch", "loss", "f1", "f2"])
            csv_file.writeheader()
            # 1 epoch is about 5 minutes roughly 12 epochs per hour
            for epoch in range(100):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                average_loss = train(my_model, train_loader, loss_function, optimizer, batch_size=16)
                epoch_f1, epoch_f2 =  validate(my_model, validate_loader, batch_size=16)
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    number_since_best = 0
                    best_epoch = epoch
                    save_model(my_model, "/dcs/large/u2102661/CS310/models/activity_recogniser/best.pt")
                else: 
                    number_since_best += 1
                save_model(my_model, "/dcs/large/u2102661/CS310/models/activity_recogniser/last.pt")
                csv_file.writerow({"epoch": epoch, "loss": average_loss, "f1": epoch_f1, "f2": epoch_f2})
                if number_since_best > 50:
                    print("Stopping early as no improvement in 50 epochs")
                    print(f"Best f1: {best_f1} at epoch {best_epoch}")
                    break
        print("Done!")

        # Print final weights of last layer
        params = list(my_model.parameters())
        print(params[-2])
        print(params[-1])


if __name__ == "__main__":
    main(is_test=True)
