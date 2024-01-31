import os
import numpy as np
import pandas as pd
import cv2
import csv
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_video, VideoReader
from torchvision.transforms import ToTensor
import torchvision.models as models


def data_preprocessing(sample): 
    """
    Input: 
        sample: A video sample from the dataset that needs to be resized to 240 x 320 pixels and 30 frames per second
    Output:
        None
    """
    cap = cv2.VideoCapture(sample)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    forcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(sample[:-4] + '_resize.mp4', forcc, 30, (320, 240))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (320, 240))
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

class AnomalyDataset(Dataset):
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
        label = self.video_labels.iloc[idx, 1]
        # Resize the video to 240 x 320 pixels 30 frames per second if it hasn't been done already
        if not os.path.exists(video_path[:-4] + "_resize.mp4"):
            data_preprocessing(video_path)
        video, audio, info = read_video(video_path[:-4] + "_resize.mp4", pts_unit='sec')
        video = video.to(dtype=torch.float32)
        sample = {'video': video, 'audio': audio, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            sample = self.target_transform(sample)
        return sample



class NormalDataset(Dataset):
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
        label = self.video_labels.iloc[idx, 1]
        # Resize the video to 240 x 320 pixels 30 frames per second if it hasn't been done already
        if not os.path.exists(video_path[:-4] + "_resize.mp4"):
            data_preprocessing(video_path)
        video, audio, info = read_video(video_path[:-4] + "_resize.mp4", pts_unit='sec')
        video = video.to(dtype=torch.float32)
        sample = {'video': video, 'audio': audio, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            sample = self.target_transform(sample)
        return sample
    


class AnomalyDetectionDataset(Dataset):
    def __init__(self, annotation_file, video_dir, transform=None, target_transform=None):

        self.video_labels = pd.read_csv(annotation_file)
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.video_labels)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 0])
        label = self.video_labels.iloc[idx, 1]
        # Resize the video to 240 x 320 pixels 30 frames per second if it hasn't been done already
        if not os.path.exists(video_path[:-4] + "_resize.mp4"):
            data_preprocessing(video_path)
        video, audio, info = read_video(video_path[:-4] + "_resize.mp4", pts_unit='sec')
        video = video.to(dtype=torch.float32)
        sample = {'video': video, 'audio': audio, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            sample = self.target_transform(sample)
        return sample


# with open("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal/labels.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["video", "label"])
#     for video in os.listdir("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal"):
#         if video.endswith(".mp4"):
#             writer.writerow([video, 0])
# with open("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly/labels.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["video", "label"])
#     for video in os.listdir("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly"):
#         if video.endswith(".mp4"):
#             writer.writerow([video, 1])
# with open("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal/labels.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["video", "label"])
#     for video in os.listdir("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal"):
#         if video.endswith(".mp4"):
#             writer.writerow([video, 0])
# with open("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly/labels.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["video", "label"])
#     for video in os.listdir("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly"):
#         if video.endswith(".mp4"):
#             writer.writerow([video, 1])
# with open("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/normal/labels.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["video", "label"])
#     for video in os.listdir("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/normal"):
#         if video.endswith(".mp4"):
#             writer.writerow([video, 0])
# with open("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/anomaly/labels.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["video", "label"])
#     for video in os.listdir("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/anomaly"):
#         if video.endswith(".mp4"):
#             writer.writerow([video, 1])

train_normal_dataset = NormalDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal",
    is_test=False)

train_anomaly_dataset = AnomalyDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly",
    is_test=False)

test_normal_dataset = NormalDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal",
    is_test=True)

test_anomaly_dataset = AnomalyDataset(
    annotation_file="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly/labels.csv",
    video_dir="/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly",
    is_test=True)

device = (
    "cuda" 
    if torch.cuda.is_available()  
    else "cpu"
)
torch.cuda.empty_cache()
print(f"Using {device} device")

train_normal_dataloader = DataLoader(train_normal_dataset, batch_size=1, shuffle=True)
train_anomaly_dataloader = DataLoader(train_anomaly_dataset, batch_size=1, shuffle=True)
test_normal_dataloader = DataLoader(test_normal_dataset, batch_size=1, shuffle=True)
test_anomaly_dataloader = DataLoader(test_anomaly_dataset, batch_size=1, shuffle=True)


class MyAnomalyDetectionModel(nn.Module):
    def __init__(self, parent_model):
        super(MyAnomalyDetectionModel, self).__init__()
        if parent_model == None:
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
        x = self.parent_model(x)
        x.squeeze_()
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = models.video.r3d_18(pretrained=True)
my_model = MyAnomalyDetectionModel(model).to(device)
my_model.parent_model.requires_grad_(False)


def loss_function(model, anomalies, normals, batch_size=32):
    lambda_temporal_smoothness = 0.001
    lambda_temporal_sparsity = 0.001
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

        loss += nn.functional.relu(1.0 - max_anomaly_score + max_normal_score) + lambda_temporal_smoothness * temporal_smoothness + lambda_temporal_sparsity * temporal_sparsity
    return loss / batch_size
# print(loss_function(my_model, [0.5, 0.1], [0.1, 0.2]))


def resize_all_clips(in_dir):
    for video in os.listdir(in_dir):
        if video.endswith(".mp4"):
            data_preprocessing(os.path.join(in_dir, video))

# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/normal")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/train/anomaly")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/normal")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/test/anomaly")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/normal")
# resize_all_clips("/dcs/large/u2102661/CS310/datasets/anomaly_detection/validation/anomaly")

loss_fun = loss_function
optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(anomaly_dataloader, normal_dataloader, model , loss_fun, optimiser, batch_size=4):
    print("Starting training")
    size = len(anomaly_dataloader)
    anomaly_scores = []
    normal_scores = []
    percentage_completes = 0
    frames_per_clip = 32
    for batch, (anomaly_features, normal_features) in enumerate(zip(anomaly_dataloader, normal_dataloader)):
        anomaly_features = anomaly_features["video"].to(device)
        normal_features = normal_features["video"].to(device)

        this_anomaly_scores = []
        # Split the anomaly clips into 16 frame clips and make predictions on each clip
        if anomaly_features.shape[1] < frames_per_clip:
            # We cannot make a fold so we just make a prediction on the whole clip
            this_anomaly_scores.append(model(anomaly_features.permute(0, 4, 1, 2, 3)))
        else:
            anomaly_clips = anomaly_features.unfold(1, frames_per_clip, frames_per_clip).permute(0, 4, 1, 5, 2, 3)
            for i in range(anomaly_clips.shape[2]):
                this_anomaly_scores.append(model(anomaly_clips[:, :, i, :, :, :]))

        # Split the normal clips into 16 frame clips and make predictions on each clip
        this_normal_scores = []
        if normal_features.shape[1] < frames_per_clip:
            # We cannot make a fold so we just make a prediction on the whole clip
            this_normal_scores.append(model(normal_features.permute(0, 4, 1, 2, 3)))
        else:
            normal_clips = normal_features.unfold(1, frames_per_clip, frames_per_clip).permute(0, 4, 1, 5, 2, 3)
            for i in range(normal_clips.shape[2]):
                this_normal_scores.append(model(normal_clips[:, :, i, :, :, :]))

        anomaly_scores.append(this_anomaly_scores)
        normal_scores.append(this_normal_scores)

        # # We need to calculate the loss on the batch
        if batch % batch_size == 0: 
            loss = loss_fun(model, anomaly_scores, normal_scores, batch_size=batch_size)
            # Backpropagate the loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Print progress
            loss, percentage_completes = loss.item(),  percentage_completes + len(anomaly_scores)
            print(f"loss: {loss:>7f}  [{percentage_completes:>5d}/{size:>5d}]")
            # Reset the lists 
            anomaly_scores = []
            normal_scores = []
    print("Finished training")

# def validate(anomaly_dataloader, normal_dataloader, model, loss_fun, optimiser): 


# def main():
#     print("Starting main")


params = list(my_model.parameters())
print(params[-1])
print(params[-2])


for epoch in range(150):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(train_anomaly_dataloader, train_normal_dataloader, my_model, loss_fun, optimiser)

params = list(my_model.parameters())
print(params[-1])
print(params[-2])

# if __name__ == "__main__":
#     main()