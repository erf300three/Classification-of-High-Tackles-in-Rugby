import os 
import sys
import numpy as np
import pandas as pd
import csv
import cv2
import random
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Creating the custom dataset
class PoseDataset(Dataset):
    def __init__(self, csv_file, number_of_keypoints, transform=None):
        self.poses = pd.read_csv(csv_file)
        self.number_of_keypoints = number_of_keypoints
        self.transform = transform
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = self.poses.iloc[idx, 0]
        frame_number = self.poses.iloc[idx, 1]
        pose_keypoints = []
        # Loop through all of the key points and add them to the pose_keypoints list
        for i in range(self.number_of_keypoints):
            pose_keypoints.append([self.poses.iloc[idx, 2 + i]])
        
        pose_keypoints = torch.tensor(pose_keypoints)
        label = self.poses.iloc[idx, 2 + self.number_of_keypoints]
        if self.transform:
            pose_keypoints = self.transform(pose_keypoints)
        
        return video_name, frame_number, pose_keypoints, label
        
        

# Create the custom classifying model
# This is a simple binary classifier with 4 fully connected layers the number 
class PoseClassifierModel(nn.Module):
    def __init__(self, number_of_key_points):
        super(PoseClassifierModel, self).__init__()
        self.fc1 = nn.Linear(number_of_key_points * 2, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = PoseDataset(csv_file="/dcs/large/u2102661/dataset/pose_classification/train.csv", number_of_keypoints=17)

validate_dataset = PoseDataset(csv_file="/dcs/large/u2102661/dataset/pose_classification/validate.csv", number_of_keypoints=17)

test_dataset = PoseDataset(csv_file="/dcs/large/u2102661/dataset/pose_classification/test.csv", number_of_keypoints=17)

def loss_function(output, target):
    class_1_weight = 1.0
    class_0_weight = 1.0
    temp1 = torch.clamp(torch.log(  output), min=-100)
    temp2 = torch.clamp(torch.log(1 - output), min=-100)

    all_loss = -(class_1_weight * (target * temp1) + class_0_weight * ((1 - target) * temp2))
    loss = torch.mean(all_loss)
    return loss

def train(model, train_loader, loss_function, optimiser, batch_size= 4):
    model.train()
    size = len(train_loader.dataset)
    current_complete = 0
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimiser.step()
        current_complete += len(X)
        print(f"Training {current_complete}/{size}", end="\r")
    print(f"Training {current_complete}/{size}")

def validate(model, train_loader, loss_function, batch_size= 4):
    model.eval()
    size = len(train_loader.dataset)
    test_loss, correct = 0, 0
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

    with torch.no_grad():
        for batch_id, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.round() == y).type(torch.float).sum().item()
            true_positive += ((pred.round() == y) & (y == 1)).type(torch.float).sum().item()
            true_negative += ((pred.round() == y) & (y == 0)).type(torch.float).sum().item()
            false_positive += ((pred.round() != y) & (y == 0)).type(torch.float).sum().item()
            false_negative += ((pred.round() != y) & (y == 1)).type(torch.float).sum().item()

    test_loss /= size
    correct /= size


    recall, precision = 0, 0
    if true_positive + false_negative != 0:
        recall = true_positive / (true_positive + false_negative)
    if true_positive + false_positive != 0:
        precision = true_positive / (true_positive + false_positive)
    f_1, f_2 = 0, 0
    if precision + recall != 0:
        f_1 = 2 * (precision * recall) / (precision + recall)
        f_2 = 5 * (precision * recall) / (4 * precision + recall)
    

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1 score: {test_loss:>8f} \n")
    return (100*correct), test_loss, recall, precision, f_1, f_2

def save_model(model, out_path):
    torch.save(model, out_path)

def main(): 
    my_model = PoseClassifierModel(17).to(device)
    optimiser = torch.optim.Adam(my_model.parameters(), lr=0.001)
    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    epochs = 10
    best_f1_score = 0
    best_epoch = 0
    number_since_best = 0
    patience = 20
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train(my_model, train_loader, loss_function, optimiser, batch_size)
        accuracy, test_loss, recall, precision, f_1, f_2 = validate(my_model, validate_loader, loss_function, batch_size)
        save_model(my_model, "/dcs/large/u2102661/models/pose_classification/last.pt")
        if f_1 > best_f1_score:
            best_f1_score = f_1
            best_epoch = e
            patience = 0
            save_model(my_model, "/dcs/large/u2102661/models/pose_classification/best.pt")
        else:
            number_since_best += 1
        if number_since_best > patience:
            print(f"Stopping early as no improvement in {patience} epochs")
            print(f"Best f1: {best_f1_score} at epoch {best_epoch}")
            break
    print("Done!")


if __name__ == "__main__":
    main()

