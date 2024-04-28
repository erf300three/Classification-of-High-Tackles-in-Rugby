"""
This file contains the code for the action recogniser model that is used to predict tackles in rugby videos
The model is based on the r3d_18 model that is pretrained on the kinetics dataset
The code contains the following classes and functions:
- ActionRecogniserDataset: This class is used to load the data from a csv file that contains the labels of the videos
    and the location of the videos
- ActionRecogniserModel: This class is used to define the model that will be used to recognise the actions in the videos
- loss_function: This function is used to calculate the loss that is used to train the model
- train: This function is used to train the model
- validate: This function is used to validate the model
- evaluate: This function is used to evaluate the model on a test set
- full_video_clips_evaluate: This function is used to evaluate the model on full video clips
- full_video_prediction: This function is used to predict the output of a video
- action_prediction: This function is used to predict the frames that are likely to contain a tackle
- write_new_films_with_predictions: This function is used to write the predictions to a new video
- resize_video: This function is used to resize a video
- get_data_object: This function is used to get the data object from a csv file
"""

import os
import csv
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.io import read_video, VideoReader
from torchvision.transforms import Normalize, Resize, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, v2

class ActionRecogniserDataset(Dataset):
    """
    This class is used to load the data from a csv file that contains the labels of the videos. It inherits from the
    torch Dataset class and overrides the __len__ and __getitem__ methods. The __len__ method returns the length of the
    dataset and the __getitem__ method returns the video, label and video name at the index that is passed in.

    Inputs:
        annotation_file: The path to the csv file that contains the labels of the videos
        video_dir: The directory that contains the videos
        is_test: A boolean that determines whether the dataset is a test dataset
        transform: The transform that is applied to the video
        target_transform: The transform that is applied to the label
    """
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
        video, _, _ = read_video(video_path, pts_unit='sec')
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
    """
    This class is used to load the data from a csv file for a full video clip. It inherits from the torch Dataset class
    and overrides the __len__ and __getitem__ methods. The __len__ method returns the length of the dataset and the
    __getitem__ method returns the video, label and video name at the index that is passed in.

    Inputs:
        annotation_file: The path to the csv file that contains the labels of the videos
        video_dir: The directory that contains the videos
        transform: The transform that is applied to the video
        target_transform: The transform that is applied to the label
    """
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
        video, _, _ = read_video(video_path, pts_unit='sec')
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
# This flatten method needs to go before the final linear layer of the model. To fix this we will manually insert the
# flatten layer into the model
# GitHub Issue: https://github.com/pytorch/vision/issues/4083

class ActionRecogniserModel(nn.Module):
    """
    This class is used to define the model that will be used to recognise the actions in the videos. It inherits from
    the torch nn.Module class and overrides the forward method. The forward method takes in the input tensor and passes
    it through the model and returns the output tensor.

    Architecture:
        - A parent model that is used to extract the features from the video based on the r3d_18 or r2plus1d_18 model
        - A flatten layer that is used to flatten the output of the parent model
        - 5 fully connected layers that are used to classify the video
        - 5 batch normalisation layers that are used to normalise the output of the fully connected layers
    
    Inputs:
        parent_model: The model that is used to extract the features from the video
        parent_model_name: The name of the parent model that is used to extract the features from the video
    
    Features:
        - Parent model: r3d_18 or r2plus1d_18
        - Flatten layer: Flattens the output of the parent model
        - Batch normalisation layers: Normalises the output of the fully connected layers
        - Dropout layer: Used to prevent overfitting with probability 0.3
        - Maxpool layers: Used to downsample the output of the fully connected layers in skip connections
        - Skip connections: Used to pass the output of the fully connected layers to the next layer
        - ReLU function: Used to introduce non-linearity to the model
        - Sigmoid function: Used to squash the output of the model to be between 0 and 1
    """
    def __init__(self, parent_model=None, parent_model_name=""):
        super(ActionRecogniserModel, self).__init__()
        # We are selecting between getting a new parent model or the parent model that is passed in
        if parent_model is None:
            self.parent_model = nn.Sequential(
                *list(models.video.r3d_18(pretrained=True).children())[:-1],
                nn.Flatten(1),
                list(models.video.r3d_18(pretrained=True).children())[-1]
            )
        elif parent_model_name == "r3d_18":
            self.parent_model = nn.Sequential(
                *list(models.video.r3d_18(pretrained=True).children())[:-1],
                nn.Flatten(1)
            )
        elif parent_model_name == "r2plus1d_18":
            self.parent_model = nn.Sequential(
                *list(models.video.r2plus1d_18(pretrained=True).children())[:-1],
                nn.Flatten(1)
            )
        else:
            self.parent_model = nn.Sequential(*list(parent_model.children()))
        self.fc0 = nn.Linear(512, 400)
        self.batch_norm_1 = nn.BatchNorm1d(400)
        self.fc1 = nn.Linear(400, 200)
        self.batch_norm_2 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.batch_norm_3 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 50)
        self.batch_norm_4 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 25)
        self.batch_norm_5 = nn.BatchNorm1d(25)
        self.fc5 = nn.Linear(25, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool16th = nn.MaxPool1d(16)
        self.maxpool4th = nn.MaxPool1d(4)

    def forward(self, x):
        """
        This function completes forward propagation through the model. It takes in the input tensor and passes it
        through the model and returns the output tensor.

        Inputs:
            x: The input tensor that is passed through the model which represents a short clip of a video
        
        Outputs:
            x: The output tensor that is returned by the model which represents the probability of a tackle in the video
        """
        x = self.parent_model(x)
        x.squeeze_()
        x = self.relu(self.fc0(x))
        x = self.batch_norm_1(x)
        residual_1 = self.maxpool16th(x)
        x = self.relu(self.fc1(x))
        x = self.batch_norm_2(x)
        x = self.dropout(x)
        residual_2 = self.maxpool4th(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm_3(x)
        x = self.dropout(x)
        residual_3 = x
        x = self.relu(self.fc3(x +residual_3 ))
        x = self.batch_norm_4(x)
        x = self.dropout(x)
        x = self.relu(self.fc4(x +residual_2))
        x = self.batch_norm_5(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc5(x + residual_1))
        return x

# Transformations to be completed on the data

# We have added normalisation to the model so we need to normalise the data before we pass it to the model
# This is with values that are outlined in the documentation for the kinetics dataset found here
# https://github.com/pytorch/vision/blob/master/references/video_classification/train.py#L115
transform_train = Compose([
    (lambda x: x / 255),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    RandomRotation(degrees=15),
    RandomHorizontalFlip(p=0.5) #,
    # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])
transform_predict = Compose([
    (lambda x: x / 255),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])



"""
=============================================================================
Calculate the class weights for the loss function when clip size = 16
=============================================================================
Class 0: Non-tackle
Class 1: Tackle

Location    Number of tackle clips    Number of non-tackle clips    Total number of clips
All dirs:   947                       19786                         20733
Train:      750                       15721                         16471
Test:       110                       2077                          2187
Validation: 87                        1988                          2075

weight_class_i = #total_clips / (#classes * #clips_in_class_i)

All dirs:   20733 / (2 * 947) = 10.95 , 20733 / (2 * 19786) = 0.5239   
Train:      16471 / (2 * 750) = 10.98 , 16471 / (2 * 15721) = 0.5239
Test:       2187 / (2 * 110) = 9.94   , 2187 / (2 * 2077) = 0.5265
Validation: 2075 / (2 * 87) = 11.93 , 2075 / (2 * 1988) = 0.5219

=============================================================================
Calculate the class weights for the loss function when clip size = 5
=============================================================================
Class 0: Non-tackle
Class 1: Tackle

Location    Number of tackle clips    Number of non-tackle clips    Total number of clips
All dirs:   2980                      63725                         66705
Train:      2358                      51058                         53416
Test:       299                       6352                          6651
Validation: 323                       6315                          6638

weight_class_i = #total_clips / (#classes * #clips_in_class_i)

All dirs:   66705 / (2 * 2980) = 11.192 , 66705 / (2 * 63725) = 0.52338   
Train:      53416 / (2 * 2358) = 11.327 , 53416 / (2 * 51058) = 0.52309
Test:       6651 / (2 * 299)   = 11.122 , 6651 / (2 * 6352)   = 0.52354
Validation: 6638 / (2 * 323)   = 10.276 , 6638 / (2 * 6315)   = 0.52557

We are going to follow the same intuition that pytorch uses for the BCELoss function
Since log(0) = -inf as lim x->0 log(x) = -inf and then we could be multiplying by 0 we can be 
in a situation where we get a loss value of nan and a gradient of lim x-> 0 ( d/dx(log(x)) = inf
which would make the backward method non linear. 

To avoid this problem we are going to do what clamp the log value to be greater than -100 
"""
def loss_function(output, target):
    """
    This function is used to calculate the loss that is used to train the model. The loss function is the binary
    cross entropy loss function with class weights. The class weights are calculated based on the number of clips in
    each class. The class weights can also be found in the comments above. We had to implement our own loss function
    as the BCELoss function in pytorch does not allow for class weights to be passed in our way. 

    We had to clamp the log values to be greater than -100 to avoid the problem of the log function going to -inf
    when the value is 0. This would cause the loss to be nan.

    Inputs:
        output: The output tensor of the model
        target: The target tensor of the model
    
    Outputs:
        loss: The loss that is calculated by the loss function stored in a tensor
    """
    class_0_weight = 0.52338
    class_1_weight = 11.192
    temp1 = torch.clamp(torch.log(output), min=-100)
    temp2 = torch.clamp(torch.log(1 - output), min=-100)
    all_loss = -(class_1_weight * (target * temp1) + class_0_weight * ((1 - target) * temp2))
    loss = torch.mean(all_loss)
    return loss

def train(model, train_loader, loss_fun, optimizer, device):
    """
    This function is used to train the model. It takes in the model, the train loader, the loss function, the optimizer
    and the device. The function will train the model on the train loader and return the mean loss of the model.
    One single call to this function will train the model for one epoch.
    
    Inputs:
        model: The model that will be trained
        train_loader: The data loader that contains the training data
        loss_fun: The loss function that will be used to calculate the loss
        optimizer: The optimizer that will be used to optimise the model
        device: The device that the model will be trained on
    """
    model.train()
    size = len(train_loader.dataset)
    computed_so_far = 0
    mean_loss = []
    print("Training...")
    for _, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        computed_so_far += len(data)
        target = target.unsqueeze(1).to(dtype=torch.float32)
        output = model(data)
        loss = loss_fun(output, target, model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss.append(loss.item())
        loss = loss.item()
        print(f"loss: {loss:>7f} batch size: {len(data)}  [{computed_so_far:>5d}/{size:>5d}]")
    mean_loss = np.mean(mean_loss)
    print(f"Mean loss: {mean_loss:>7f}")
    return mean_loss

def validate(model, valid_loader, device):
    """
    This function is used to validate the model. It takes in the model, the valid loader and the device. The function
    will validate the model on the vaildation set and return the f1 score and f2 score of the model. One single call to
    this function will validate the model on the validation set. 

    Inputs:
        model: The model that will be validated
        valid_loader: The data loader that contains the validation data
        device: The device that the model will be validated on
    """
    model.eval()
    print("Validating...")
    size = len(valid_loader.dataset)
    computed_so_far = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    # For each batch in the test set we need to calculate the TP, TN, FP, FN
    with torch.no_grad():
        for _, (data, target, _) in enumerate(valid_loader):
            computed_so_far += len(data)
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1).to(dtype=torch.float32)
            output = model(data)
            batch_tp = 0
            batch_tn = 0
            batch_fp = 0
            batch_fn = 0

            # Calculate the TP, TN, FP, FN for this batch
            for element in zip(output, target): # Element 0 is the output, element 1 is the target
                if element[0] >= 0.5 and element[1] == 1:
                    true_positives += 1
                    batch_tp += 1
                elif element[0] >= 0.5 and element[1] == 0:
                    false_positives += 1
                    batch_fp += 1
                elif element[0] < 0.5 and element[1] == 1:
                    false_negatives += 1
                    batch_fn += 1
                elif element[0] < 0.5 and element[1] == 0:
                    true_negatives += 1
                    batch_tn += 1
            # We could have the case where there are no positive clips in the batch
            if batch_tp + batch_fn== 0:
                batch_f1_score = 0
                print(f"batch f1 score: N/A  batch size: {len(data)}  [{computed_so_far:>5d}/{size:>5d}]")
            elif batch_tp + batch_fp == 0:
                batch_f1_score = 0
                print(f"batch f1 score: N/A  batch size: {len(data)}  [{computed_so_far:>5d}/{size:>5d}]")
            else:
                recall = batch_tp / (batch_tp + batch_fn)
                precision = batch_tp / (batch_tp + batch_fp)
                batch_f1_score = 0
                if recall + precision != 0:
                    batch_f1_score = 2 * (precision * recall / (precision + recall))
                print(f"recall: {recall:>7f} precision: {precision:>7f}")
                print(f"batch f1 score: {batch_f1_score:>7f} batch size: {len(data)} [{computed_so_far:>5d}/{size:>5d}]")

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
    """
    This function is used to resize a video. It takes in the input tensor and the size that the video will be resized to
    and returns the resized video.

    Inputs:
        input_tensor: The input tensor that represents the video
        size: The size that the video will be resized to
    
    Outputs:
        resized_video: The resized video
    """
    return torch.stack([v2.Resize(size)(input_tensor[i, :, :, :, :]) for i in range(input_tensor.shape[0])], dim=0)

def evaluate(model, test_loader, device, output_path):
    """
    This function is used to evaluate the model on a test set. It takes in the model, the test loader, the device. 
    The function will evaluate the model on the test set and return the f1 score and f2 score of the model. One single
    call to this function will evaluate the model on the test set.

    Inputs:
        model: The model that will be evaluated
        test_loader: The data loader that contains the test data
        device: The device that the model will be evaluated on
        output_path: The path where the output will be saved
    
    Outputs:
        The results of the evaluation will be saved in a csv file in the output path
    """
    model.eval()
    print("Evaluating...")
    size = len(test_loader)
    number_completed = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    with open(os.path.join(output_path, "evaluation.csv"), 'w', newline='', encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["video_name", "output", "target"])
        csv_file.writeheader()
        with torch.no_grad():
            for _, (data, target, video_name) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                target = target.unsqueeze(1).to(dtype=torch.float32)
                output = model(data)
                for element in zip(output, target): # Element 0 is the output, element 1 is the target
                    if element[0] >= 0.5 and element[1] == 1:
                        true_positives += 1
                    elif element[0] >= 0.5 and element[1] == 0:
                        false_positives += 1
                    elif element[0] < 0.5 and element[1] == 1:
                        false_negatives += 1
                    elif element[0] < 0.5 and element[1] == 0:
                        true_negatives += 1
                number_completed += len(data)
                for element in zip(video_name, output, target):
                    # Element 0 is the video name, element 1 is the output, element 2 is the target
                    csv_file.writerow({"video_name":element[0], "output":element[1].item(), "target":element[2].item()})
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
    field_names= [
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "recall",
        "precision",
        "f1_score",
        "f2_score"
    ]
    with open(os.path.join(output_path, "results.csv"), 'w', newline='', encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=field_names)
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

def full_video_clips_evaluate(model, dir_to_test, output_path):
    """
    This function is used to evaluate the model on full video clips. It takes in the model, the directory to test and
    the output path. The function will evaluate the model on the full video clips in the test directory and return the
    results of the evaluation. One single call to this function will evaluate the model on all the full video clips in 
    the test directory.

    Inputs:
        model: The model that will be evaluated
        dir_to_test: The directory that contains the full video clips
        output_path: The path where the output will be saved
    
    Outputs:
        The results of the evaluation will be saved in a csv file in the output path
    """
    model.eval()
    print("Testing full video clips...")

    full_length_clip_transform = Compose([
        (lambda x: x / 255),
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989], inplace=True)
    ])

    def transform_csv_to_object(in_path):
        output_file = os.path.join(in_path, "tackles.csv")
        data = {}
        with open(output_file, 'r', newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["video_name"] not in data:
                    data[row["video_name"]] = []
                data[row["video_name"]].append((int(row["start_frame"]), int(row["end_frame"])))
        return data

    tackle_data = transform_csv_to_object(dir_to_test)
    with open(os.path.join(output_path, "full_clips_evaluation.csv"), 'w', newline='', encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["video_name", "frame_number", "output"])
        csv_file.writeheader()
        for video_name in tackle_data:
            video_path = os.path.join(dir_to_test, video_name)
            # Read the video and split it into 16 frame clips
            video, _, _ = read_video(video_path, pts_unit='sec')
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
                    concat_videos = torch.stack([video[:, i + j, :, :, :] for j in range(0, video.shape[1] - i)], dim=0)
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
                    for idx, item in enumerate(output):
                        csv_file.writerow({"video_name": video_name, "frame_number": i + idx, "output": item.item()})
                del transformed_video
                del output
                print(torch.cuda.memory_allocated() / 1024 ** 3)

            del video
    print("Done testing full video clips")

def full_video_prediction(model, video_path, csv_path, is_full_version=True, frame_length=16):
    """
    This function is used to make predictions on the frames of a video. This function reads the videos segement 
    by segeemnt and makes predictions on the frames in that segment. The function will then write all of these
    predictions to a csv file in the csv path. The function will return the predictions that were made.

    Inputs:
        model: The model that will be used to make the predictions
        video_path: The path to the video that the predictions will be made on
        csv_path: The path where the predictions will be written
        is_full_version: A boolean that determines whether the model has to be run or just the csv file has to be read
        frame_length: The length of the frame that the model will be predicting on
    
    Outputs:
        outputs: The predictions that were made by the model for the sliding windows with stride 1
    """
    outputs = []
    clip = None
    clips = []
     # This reader is just used to get the metadata of the video so we can iterate through the frames correctly
    reader = VideoReader(video_path, "video")
    fps = reader.get_metadata()["video"]["fps"][0]
    total_frames = int(fps * reader.get_metadata()["video"]["duration"][0])
    if not is_full_version:
        with open(csv_path, 'r', newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            outputs = []
            for row in reader:
                outputs.append((int(row["frame_number"]), float(row["output"])))
        return outputs
    if frame_length == 16:
        for i in range(8, total_frames - 8):
            print(f"Predicting frame {i} of {total_frames}")
            clip = None
            clip = read_video(video_path, pts_unit='sec', start_pts=(i-8) * float(1/fps), end_pts=(i+7) * float(1/fps))
            if clip[0].shape[0] > 16:
                clip = clip[0][-16:, :, :, :]
            elif clip[0].shape[0] < 16:
                temp_tensor = torch.zeros((16 - clip[0].shape[0], clip[0].shape[1], clip[0].shape[2], clip[0].shape[3]))
                clip = torch.cat((temp_tensor, clip[0]), dim=0)
            else:
                clip = clip[0]
            clip = clip.permute(0, 3, 1, 2)
            clip = clip.to(device=device, dtype=torch.float32)
            clip = transform_predict(clip)
            clip = Resize((224, 224))(clip)
            clips.append(clip.permute(1, 0, 2, 3))
            if i % 32 == 7 or i == total_frames - 9:
                temp_stack = torch.stack(clips, dim=0)
                for clip in clips:
                    del clip
                clips = temp_stack
                with torch.no_grad():
                    output = model(clips)
                clips = []
                for idx, item in enumerate(output):
                    outputs.append((i - 32 + idx + 1, item.item()))
    elif frame_length == 5:
        for i in range(2, total_frames - 3):
            print(f"Predicting frame {i} of {total_frames}")
            clip = None
            clip = read_video(video_path, pts_unit='sec', start_pts=(i-2) * float(1/fps), end_pts=(i+2) * float(1/fps))
            if clip[0].shape[0] > 5:
                clip = clip[0][-5:, :, :, :]
            elif clip[0].shape[0] < 5:
                temp_tensor = torch.zeros((5 - clip[0].shape[0], clip[0].shape[1], clip[0].shape[2], clip[0].shape[3]))
                clip = torch.cat((temp_tensor, clip[0]), dim=0)
            else:
                clip = clip[0]
            clip = clip.permute(0, 3, 1, 2)
            clip = clip.to(device=device, dtype=torch.float32)
            clip = transform_predict(clip)
            clip = Resize((224, 224))(clip)
            clips.append(clip.permute(1, 0, 2, 3))
            if i % 128 == 2 or i == total_frames - 3:
                temp_stack = torch.stack(clips, dim=0)
                for clip in clips:
                    del clip
                clips = temp_stack
                with torch.no_grad():
                    output = model(clips)
                clips = []
                for idx, item in enumerate(output):
                    outputs.append((i - 128 + idx + 1, item.item()))
    with open(csv_path, 'w', newline='', encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["frame_number", "output"])
        csv_file.writeheader()
        for output in outputs:
            csv_file.writerow({"frame_number": output[0], "output": output[1]})

def action_prediction(
        model,
        video_path,
        output_dir,
        is_full_version=True,
        confidence_threshold=0.5,
        frame_length=16,
        group_size=10,
        group_required=4
):
    """
    This function is used within the pipeline to make predictions on the frames of a video that are likely to contain 
    a tackle. The function will return the frame numbers of the frames to create a group from, in which the group
    is likely to contain a tackle. All of the predictions will be written to a csv file in the output directory
    meaning this stage can be skipped if the predictions have already been made. Allowing for iterations to be completed
    without having to re-run the model.

    Inputs:
        model: The model that will be used to make the predictions
        video_path: The path to the video that the predictions will be made on
        output_dir: The directory where the csv file containing the predictions will be written
        is_full_version: A boolean that determines whether the model has to be run or just the csv file has to be read
        confidence_threshold: The threshold that the model has to pass to be considered a tackle
        frame_length: The number of frames that the model will be predicting on
        group_size: The maximum distance a frame can be from another frame to be considered in the same group
        group_required: The minimum number of frames that have to be in a group to be considered a tackle

    Outputs:
        A list of frame numbers to centre a clip around that are likely to contain a tackle
    """
    print("Predicting frames likely to contain a tackle...")
    outputs = []
    video_name = video_path.split(os.sep)[-1][:-4]
    outputs = full_video_prediction(
        model=model,
        video_path=video_path,
        csv_path=os.path.join(output_dir, "action_recogniser", video_name + ".csv"),
        is_full_version=is_full_version,
        frame_length=frame_length
    )
    # We now want to return frames where there are at least the group required number of frames within the group size
    # that have a confidence level of at least the confidence threshold
    # We are using a nested function since no other function will need to use this and it makes the code more readable
    def group_frames(numbers, max_difference=group_size):
        groups = []
        for number in numbers:
            found_group = False
            for group in groups:
                for member in group:
                    if abs(member[0] - number[0]) <= max_difference:
                        group.append(number)
                        found_group = True
                        break
                    # remove this if-block if a number should be added to multiple groups
                    if found_group:
                        break
            if not found_group:
                groups.append([number])
        return groups

    confident_frames = [outputs[i] for i in range(len(outputs)) if outputs[i][1] >= confidence_threshold]
    groups = group_frames(confident_frames)
    filtered_groups = [group for group in groups if len(group) >= group_required]
    # Return the frame number of highest confidence in each group
    results = [max(group, key=lambda x: x[1]) for group in filtered_groups]
    return [result[0] for result in results]

def write_new_films_with_predictions(eval_csv, video_dir):
    """
    This function is used to write new videos with the predictions of the model displayed on the video. The function
    reads the predictions from a csv file and uses them to write the new video. 

    Inputs:
        eval_csv: The path to the csv file that contains the predictions
        video_dir: The directory that contains the videos that the predictions were made on
    
    Outputs:
        The new videos with the predictions will be written to the directory that contains the videos
    """
    def get_data_object(in_path):
        output_file = in_path
        data = {}
        with open(output_file, 'r', newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["video_name"] not in data:
                    data[row["video_name"]] = []
                data[row["video_name"]].append((int(row["frame_number"]), float(row["output"])))
        return data

    eval_data = get_data_object(eval_csv)

    print("Writing new videos with predictions...")

    for data in eval_data.items():
        video_name = data[0]
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
                    frame = cv2.putText(
                        frame,
                        "Tackle prediction: " + str(output),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                elif output < 0.5:
                    frame = cv2.putText(
                        frame,
                        "Tackle prediction: " + str(output),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
            out.write(frame)
        cv2_video.release()
        out.release()
    cv2.destroyAllWindows()

def save_model(model, path):
    """
    This function is used to save the model to a file. The model will be saved to the path that is passed in.

    Inputs:
        model: The model that will be saved
        path: The path where the model will be saved
    
    Outputs:
        The model will be saved to the path that is passed in
    """
    torch.save(model, path)

def main():
    """
    This function is the main function that parses theh system arguments and runs the model. The function will
    mostly be used to train the model but can also be used to test the model. 

    Arguments:
        --is_test: A boolean that determines whether the model is being tested
        --parent_model_name: The name of the parent model that the action recogniser is based on
        --op: The operation that the model will be performing
        --frame_length: The length of the frame that the model will be predicting on
        --group_size: The maximum distance a frame can be from another frame to be considered in the same group
        --group_required: The minimum number of frames that have to be in a group to be considered a tackle
        --confidence_threshold: The threshold that the model has to pass to be considered a tackle
    """

    parser = argparse.ArgumentParser(description="Train an action recogniser model")
    parser.add_argument("--is_test", type=bool, default=False, help="Whether the model is being tested")
    parser.add_argument("--parent_model_name", type=str, default="r3d_18", help="The parent model that the action recogniser is based on")
    parser.add_argument("--op", type=str, default="train", help="The operation that the model will be performing")
    parser.add_argument("--frame_length", type=int, default=16, help="The length of the frame that the model will be predicting on")
    parser.add_argument("--group_size", type=int, default=10, help="The maximum distance a frame can be from another frame to be considered in the same group")
    parser.add_argument("--group_required", type=int, default=4, help="The minimum number of frames that have to be in a group to be considered a tackle")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="The threshold that the model has to pass to be considered a tackle")
    parser.add_argument("--model_path", type=str, help="The path to the model")
    # default /dcs/large/u2102661/CS310/models/activity_recogniser/best.pt
    parser.add_argument("--out_path", type=str, help="The path to the output directory")
    # default /dcs/large/u2102661/CS310/model_evaluation/action_recogniser/
    args = parser.parse_args()
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the data
    if args.frame_length == 16:
        data_dir = "/dcs/large/u2102661/CS310/datasets/activity_recogniser"
    elif args.frame_length == 5:
        data_dir = "/dcs/large/u2102661/CS310/datasets/activity_recogniser_5_frames"

    train_data = ActionRecogniserDataset(
        annotation_file=os.path.join(data_dir, "train", "labels.csv"),
        video_dir=os.path.join(data_dir, "train"),
        is_test=False,
        transform=transform_train
    )
    test_data = ActionRecogniserDataset(
        annotation_file=os.path.join(data_dir, "test", "labels.csv"),
        video_dir=os.path.join(data_dir, "test"),
        is_test=True,
        transform=transform_predict
    )
    validate_data = ActionRecogniserDataset(
        annotation_file=os.path.join(data_dir, "validation", "labels.csv"),
        video_dir=os.path.join(data_dir, "validation"),
        is_test=True,
        transform=transform_predict
    )

    # Create the data loaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)
    validate_loader = DataLoader(validate_data, batch_size=128, shuffle=True)

    match args.op:
        case "test":
            my_model = torch.load(args.model_path, map_location=device)
            my_model = my_model.to(device)
            evaluate(my_model, test_loader, device, "/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/")
        case "full_clips_test":
            my_model = torch.load(args.model_path, map_location=device)
            my_model = my_model.to(device)
            full_video_clips_evaluate(
                my_model,
                "/dcs/large/u2102661/CS310/datasets/activity_recogniser/full_length_clips/",
                args.out_path
            )
            write_new_films_with_predictions(
                "/dcs/large/u2102661/CS310/model_evaluation/action_recogniser/full_video_clips.csv",
                "/dcs/large/u2102661/CS310/datasets/activity_recogniser/full_length_clips"
            )
        case "train":
            if args.parent_model_name == "r3d_18":
                model = models.video.r3d_18(pretrained=True)
                my_model = ActionRecogniserModel(model, "r3d_18").to(device)
            elif args.parent_model_name == "r2plus1d_18":
                model = models.video.r2plus1d_18(pretrained=True)
                my_model = ActionRecogniserModel(model, "r2plus1d_18").to(device)
            my_model.parent_model.requires_grad_(False)
            best_f1 = -1
            number_since_best = 0
            best_epoch = 0
            patience = 30
            optimizer = torch.optim.Adam(my_model.parameters(), lr = 0.001)
            with open(os.path.join(args.out_path, "loss.csv"), 'w', newline='', encoding="utf-8") as csvfile:
                csv_file = csv.DictWriter(csvfile, fieldnames=["epoch", "loss", "f1", "f2"])
                csv_file.writeheader()
                for epoch in range(100):
                    print(f"Epoch {epoch + 1}\n-------------------------------")
                    average_loss = train(my_model, train_loader, loss_function, optimizer, device)
                    epoch_f1, epoch_f2 =  validate(my_model, validate_loader, device)
                    if epoch_f1 > best_f1:
                        best_f1 = epoch_f1
                        number_since_best = 0
                        best_epoch = epoch
                        save_model(my_model, os.path.join(args.out_path, "best.pt"))
                    else:
                        number_since_best += 1
                    save_model(my_model, os.path.join(args.out_path, "last.pt"))
                    csv_file.writerow({"epoch": epoch, "loss": average_loss, "f1": epoch_f1, "f2": epoch_f2})
                    if number_since_best > patience:
                        print("Stopping early as no improvement in 30 epochs")
                        print(f"Best f1: {best_f1} at epoch {best_epoch}")
                        break
            print(f"Best f1: {best_f1} at epoch {best_epoch}")
            print("Done!")

if __name__ == "__main__":
    main()
