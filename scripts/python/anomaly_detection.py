"""
This file contains all of the code related to the anoly detection model. This model was trained and evaluated but found
to be worse than the action recognition and was not used in the final pipeline. This model was not mentioned in the
report due to word count constraints. The code is left here for reference.
"""
import os
import sys
import csv
import pandas as pd
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_video
from torchvision.transforms import Normalize, Compose, RandomRotation, RandomHorizontalFlip
import torchvision.models as models

def data_preprocessing(sample):
    """
    This function resizes the video to 240 x 320 pixels and 30 frames per second and saves it to a new file

    Input: 
        sample: path to the video file to resize
    
    Output:
        A resized video file with the same name as the input file but with _resize appended to the end
    """
    cap = cv2.VideoCapture(sample)
    forcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(sample[:-4] + '_resize.mp4', forcc, 30, (320, 240))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

class AnomalyDataset(Dataset):
    """
    This class is a dataset class for anomaly class of the anomaly detection model. It reads the video files and 
    their labels from the annotation file and returns the video and the label when called. The video is resized 
    to 240 x 320 pixels and 30 frames per second if it hasn't been done already.
    
    Inputs:
        annotation_file: path to the csv file containing the video file names and their labels
        video_dir: path to the directory containing the video files
        is_test: boolean to indicate if the dataset is a test dataset
        transform: a transformation to apply to the video
        target_transform: a transformation to apply to the label
    """
    def __init__(self, annotation_file, video_dir, is_test=False, transform=None, target_transform=None):
        self.video_labels = pd.read_csv(annotation_file)
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 0])
        video_name = self.video_labels.iloc[idx, 0]
        label = self.video_labels.iloc[idx, 1]
        # Resize the video to 240 x 320 pixels 30 frames per second if it hasn't been done already
        if not os.path.exists(video_path[:-4] + "_resize.mp4"):
            data_preprocessing(video_path)
        video, _, _ = read_video(video_path[:-4] + "_resize.mp4", pts_unit='sec')
        video = video.permute(0, 3, 1, 2)
        video = video.to(dtype=torch.float32)
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label, video_name

class NormalDataset(Dataset):
    """
    This class is a dataset class for normal class of the anomaly detection model. It reads the video files and 
    their labels from the annotation file and returns the video and the label when called. The video is resized
    to 240 x 320 pixels and 30 frames per second if it hasn't been done already.

    Inputs:
        annotation_file: path to the csv file containing the video file names and their labels
        video_dir: path to the directory containing the video files
        is_test: boolean to indicate if the dataset is a test dataset
        transform: a transformation to apply to the video
        target_transform: a transformation to apply to the label
    """
    def __init__(self, annotation_file, video_dir, is_test=False, transform=None, target_transform=None):
        self.video_labels = pd.read_csv(annotation_file)
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 0])
        video_name = self.video_labels.iloc[idx, 0]
        label = self.video_labels.iloc[idx, 1]
        # Resize the video to 240 x 320 pixels 30 frames per second if it hasn't been done already
        if not os.path.exists(video_path[:-4] + "_resize.mp4"):
            data_preprocessing(video_path)
        video, _, _ = read_video(video_path[:-4] + "_resize.mp4", pts_unit='sec')
        video = video.permute(0, 3, 1, 2)
        video = video.to(dtype=torch.float32)
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label, video_name

class MyAnomalyDetectionModel(nn.Module):
    """
    This class is the anomaly detection model. It is a combination of the R3D_18 model and a custom model that
    takes the output of the R3D_18 model and passes it through a fully connected network to get the anomaly score.

    Inputs:
        parent_model: the parent model to use. If None, the R3D_18 model is used
    
    Architecture:
        - The parent model is the R3D_18 model with the last layer removed
        - A fully connected layer with 512 input features and 256 output features
        - A fully connected layer with 256 input features and 32 output features
        - A fully connected layer with 32 input features and 1 output feature

    Features:
        - ReLU activation function
        - Dropout with a probability of 0.6
        - Sigmoid activation function
    """
    def __init__(self, parent_model):
        super(MyAnomalyDetectionModel, self).__init__()
        if parent_model is None:
            # Need to implement this for C3d model for the moment it can be r3d_18
            self.parent_model = nn.Sequential(*list(models.video.r3d_18(pretrained=True).children())[:-1])
        else:
            # Load the parent model and remove the last layer (classsification)
            self.parent_model = nn.Sequential(*list(parent_model.children())[:-1])
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        This function is the forward pass of the model. It takes the input and passes it through the parent model
        and then through the fully connected layers to get the anomaly score. It takes an input tensor and
        returns a tensor with the anomaly score.

        Inputs:
            x: input tensor representing a segment of the video clip
        
        Output:
            x: tensor with the anomaly score
        """
        x = self.parent_model(x)
        x.squeeze_()
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def loss_function(model, anomalies, normals, batch_size=32):
    """
    This function calculates the loss for the model for a prediction on a batch of anomaly and normal clips.
    The loss is a combination of the following:
        1. The hinge loss for the max anomaly score and the max normal score
        2. The temporal smoothness of the anomaly scores
        3. The temporal sparsity of the anomaly scores
        4. The matrix norm of the model parameters (only the layers that require a gradient)
    With all of that a single tensor containing the loss for the batch is returned.

    Inputs:
        model: the model to calculate the loss for
        anomalies: a list of lists containing the anomaly scores for each clip in the batch
        normals: a list of lists containing the normal scores for each clip in the batch
        batch_size: the size of the batch
    
    Output:
        loss: a tensor containing the loss for the batch
    """
    lambda_temporal_smoothness = 0.001
    lambda_temporal_sparsity = 0.001
    # The matrix norm has not been implemented yet
    lambda_matrix_norm = 0.001
    loss = torch.tensor([0.0]).cuda()
    for i in range(batch_size):
        anomaly_index = torch.randint(0, len(anomalies), (1,)).cuda()
        normal_index = torch.randint(0, len(normals), (1,)).cuda()

        anomaly_bag = anomalies[anomaly_index]
        normal_bag = normals[normal_index]
        max_anomaly_score = max(anomaly_bag)
        max_normal_score = max(normal_bag)
        # Calculate the temporal sparsity
        temporal_sparsity = sum(anomaly_bag)
        # Calculate the temporal smoothness
        temp = []
        for i in range(len(anomaly_bag) - 1):
            temp.append(torch.pow(anomaly_bag[i] - anomaly_bag[i + 1], 2))
        temporal_smoothness = sum(temp)
        # Calculate the matrix norm for everything that requires a gradient
        for param in model.parameters():
            if param.requires_grad:
                loss += torch.norm(param, p=2)

        loss += nn.functional.relu(1.0 - max_anomaly_score + max_normal_score)  \
                + lambda_temporal_smoothness * temporal_smoothness \
                + lambda_temporal_sparsity * temporal_sparsity
    return loss / batch_size

def resize_all_clips(in_dir):
    """
    Function to resize all of the clips in the input directory

    Inputs:
        in_dir: path to the directory containing the clips to resize
    
    Output:
        Resized clips in the same directory with _resize appended to the end of the file name
    """
    for video in os.listdir(in_dir):
        if video.endswith(".mp4"):
            data_preprocessing(os.path.join(in_dir, video))

# We initially resized all of the clips in the dataset
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/normal")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/anomaly")

# All of the transformations that were required for each datapoint
transform_main = Compose([
    (lambda x: x / 255.0),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15)
])
transform_normalise = Compose([
    (lambda x: x / 255.0),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

# Create the datasets and dataloaders
train_normal_dataset = NormalDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal",
    is_test=False,
    transform=transform_main
)
train_anomaly_dataset = AnomalyDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly",
    is_test=False,
    transform=transform_main
)
validation_normal_dataset = NormalDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/normal/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/normal",
    is_test=False,
    transform=transform_normalise
)
validation_anomaly_dataset = AnomalyDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/anomaly/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/anomaly",
    is_test=False,
    transform=transform_normalise
)
test_normal_dataset = NormalDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal",
    is_test=True
)
test_anomaly_dataset = AnomalyDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly",
    is_test=True
)

train_normal_dataloader = DataLoader(train_normal_dataset, batch_size=1, shuffle=True)
train_anomaly_dataloader = DataLoader(train_anomaly_dataset, batch_size=1, shuffle=True)
validation_normal_dataloader = DataLoader(validation_normal_dataset, batch_size=1, shuffle=True)
validation_anomaly_dataloader = DataLoader(validation_anomaly_dataset, batch_size=1, shuffle=True)
test_anomaly_dataloader = DataLoader(test_anomaly_dataset, batch_size=1, shuffle=True)
test_normal_dataloader = DataLoader(test_normal_dataset, batch_size=1, shuffle=True)

# Globally used to determine what device computations are done on
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(anomaly_dataloader, normal_dataloader, model , loss_fun, optimiser, batch_size=4):
    """
    This function trains the model on the input dataloaders. It iterates through both the normal and anomaly dataloaders
    in parallel and makes predictions on each clip in the video. It then calculates the loss for the batch between 
    the anomaly and normal clips and backpropagates it. It prints the progress as it goes along. One call to this 
    function is one epoch of training.

    Inputs:
        anomaly_dataloader: the dataloader for the anomaly dataset
        normal_dataloader: the dataloader for the normal dataset
        model: the model to train
        loss_fun: the loss function to use
        optimiser: the optimiser to use
    """
    print("Starting training")
    size = len(anomaly_dataloader)
    anomaly_scores = []
    normal_scores = []
    videos_complete = 0
    frames_per_clip = 16
    for batch, (anomaly_data, normal_data) in enumerate(zip(anomaly_dataloader, normal_dataloader)):
        (anomaly_features, _, _) = anomaly_data
        anomaly_features = anomaly_features.to(device)
        (normal_features, _, _) = normal_data
        normal_features = normal_features.to(device)

        this_anomaly_scores = []
        # Split the anomaly clips into 16 frame clips and make predictions on each clip
        if anomaly_features.shape[1] < frames_per_clip:
            # We cannot make a fold so we just make a prediction on the whole clip
            this_anomaly_scores.append(model(anomaly_features.permute(0, 2, 1, 3, 4)))
        else:
            anomaly_clips = anomaly_features.unfold(1, frames_per_clip, frames_per_clip).permute(0, 2, 1, 5, 3, 4)

            for i in range(anomaly_clips.shape[2]):
                this_anomaly_scores.append(model(anomaly_clips[:, :, i, :, :, :]))

        # Split the normal clips into 16 frame clips and make predictions on each clip
        this_normal_scores = []
        if normal_features.shape[1] < frames_per_clip:
            # We cannot make a fold so we just make a prediction on the whole clip
            this_normal_scores.append(model(normal_features.permute(0, 2, 1, 3, 4)))
        else:
            normal_clips = normal_features.unfold(1, frames_per_clip, frames_per_clip).permute(0, 2, 1, 5, 3, 4)
            for i in range(normal_clips.shape[2]):
                this_normal_scores.append(model(normal_clips[:, :, i, :, :, :]))

        anomaly_scores.append(this_anomaly_scores)
        normal_scores.append(this_normal_scores)
        videos_complete += 1
        # # We need to calculate the loss on the batch
        if batch % batch_size == 0:
            loss = loss_fun(model, anomaly_scores, normal_scores, batch_size=batch_size)
            # Backpropagate the loss
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            # Print progress
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{videos_complete:>5d}/{size:>5d}]")
            # Reset the lists
            anomaly_scores = []
            normal_scores = []
    print("Finished training")

def validate(anomaly_dataloader, normal_dataloader, model, loss_fun, batch_size=4):
    """
    This function validates the model on the validation dataset. It iterates through both the normal and anomaly
    dataloaders in parallel and make predictions on each clip in the video. It then calculates the loss for the
    batch and prints the progress. Validation is done by selecting the model with the lowest loss.

    Inputs:
        anomaly_dataloader: the dataloader for the anomaly dataset
        normal_dataloader: the dataloader for the normal dataset
        model: the model to validate
        loss_fun: the loss function to use
        batch_size: the size of the batch
    
    Output:
        total_loss: the total loss for the validation dataset
    """
    model.eval()
    print("Starting validation")
    size = len(anomaly_dataloader)
    videos_complete = 0
    anomaly_scores = []
    normal_scores = []
    frames_per_clip = 16
    total_loss = 0
    print("===========================================validating===========================================")
    with torch.no_grad(): # We don't need to calculate gradients for validation
        for batch, (anomaly_data, normal_data) in enumerate(zip(anomaly_dataloader, normal_dataloader)):
            (anomaly_video, _, _) = anomaly_data
            anomaly_video = anomaly_video.to(device)
            (normal_video, _, _) = normal_data
            normal_video = normal_video.to(device)

            this_anomaly_scores = []
            # Split the anomaly clips into 16 frame clips and make predictions on each clip
            if anomaly_video.shape[1] < frames_per_clip:
                # We cannot make a fold so we just make a prediction on the whole clip
                this_anomaly_scores.append(model(anomaly_video.permute(0, 2, 1, 3, 4)))
            else:
                anomaly_clips = anomaly_video.unfold(1, frames_per_clip, frames_per_clip).permute(0, 2, 1, 5, 3, 4)
                for i in range(anomaly_clips.shape[2]):
                    this_anomaly_scores.append(model(anomaly_clips[:, :, i, :, :, :]))
            this_normal_scores = []
            # Split the normal clips into 16 frame clips and make predictions on each clip
            if normal_video.shape[1] < frames_per_clip:
                # We cannot make a fold so we just make a prediction on the whole clip
                this_normal_scores.append(model(normal_video.permute(0, 2, 1, 3, 4)))
            else:
                normal_clips = normal_video.unfold(1, frames_per_clip, frames_per_clip).permute(0, 2, 1, 5, 3, 4)
                for i in range(normal_clips.shape[2]):
                    this_normal_scores.append(model(normal_clips[:, :, i, :, :, :]))
            anomaly_scores.append(this_anomaly_scores)
            normal_scores.append(this_normal_scores)

            videos_complete += 1
            if batch % batch_size == 0:
                loss = loss_fun(model, anomaly_scores, normal_scores, batch_size=batch_size)
                # Print progress
                loss = loss.item()
                total_loss += loss
                print(f"loss: {loss:>7f} [{videos_complete:>5d}/{size:>5d}]")
                # Reset the lists
                anomaly_scores = []
                normal_scores = []
    return total_loss

def save_model(model, path):
    """
    A function to save the model to the input path

    Inputs:
        model: the model to save
        path: the path to save the model to
    
    Output:
        The model saved to the input path
    """
    torch.save(model, path)

def main():
    """
    This is the main function that is responsible for training and evaluating the anomaly detection model. 
    It comtains our training loop along with our validation to select the best epoch. 
    """
    model = models.video.r3d_18(pretrained=True)
    my_model = MyAnomalyDetectionModel(model).to(device)
    my_model.parent_model.requires_grad_(False)
    loss_fun = loss_function
    optimiser = torch.optim.Adam(my_model.parameters(), lr=0.001)

    params = list(my_model.parameters())
    print(params[-1])
    print(params[-2])

    best_total_loss = float(sys.maxsize)
    best_epoch = 0
    number_since_best = 0
    epoch_loss = 0

    # Open csv to write the loss to for each epoch
    with open("/dcs/large/u2102661/CS310/models/anomaly_detection/loss.csv","w",newline="",encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, fieldnames=["epoch", "total loss"])
        csv_file.writeheader()
        for epoch in range(150):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train(train_anomaly_dataloader, train_normal_dataloader, my_model, loss_fun, optimiser)
            epoch_loss = validate(validation_anomaly_dataloader, validation_normal_dataloader, my_model, loss_fun)
            if epoch_loss < best_total_loss:
                best_total_loss = epoch_loss
                number_since_best = 0
                best_epoch = epoch
                save_model(my_model, "/dcs/large/u2102661/CS310/models/anomaly_detection/best_model.pt")
            else:
                number_since_best += 1
            save_model(my_model, "/dcs/large/u2102661/CS310/models/anomaly_detection/last_model.pt")
            csv_file.writerow({"epoch": epoch + 1, "total loss": epoch_loss})
            if number_since_best > 20:
                print("Stopping early as no improvement in 20 epochs")
                print(f"Best total loss: {best_total_loss} at epoch {best_epoch}")
                break
    print("Done!")
    params = list(my_model.parameters())
    print(params[-1])
    print(params[-2])

if __name__ == "__main__":
    main()
