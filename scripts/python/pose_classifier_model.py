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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Creating the custom dataset
# Labels are 0 for no tackle, 1 for low tackle and 2 for high tackle
class PoseDataset(Dataset):
    def __init__(self, csv_file, number_of_keypoints, transform=None):
        self.poses = pd.read_csv(csv_file)
        # Each player has 17 keypoints, each with an x and y coordinate and there are 2 players in each tackle 
        self.number_of_keypoints = number_of_keypoints * 4
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
# This is a simple multi-class classifier with 4 fully connected layers the number 
class PoseClassifierModel(nn.Module):
    def __init__(self, number_of_key_points):
        super(PoseClassifierModel, self).__init__()
        # self.fc1 = nn.Linear(number_of_key_points * 4, 128)
        self.fc1 = nn.Linear(68, 128)
        # self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        # self.batch_norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        # self.batch_norm3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 3)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = PoseDataset(csv_file="/dcs/large/u2102661/CS310/datasets/pose_estimation/train/pose_classification_train.csv", number_of_keypoints=17)

validate_dataset = PoseDataset(csv_file="/dcs/large/u2102661/CS310/datasets/pose_estimation/valid/pose_classification_valid.csv", number_of_keypoints=17)

test_dataset = PoseDataset(csv_file="/dcs/large/u2102661/CS310/datasets/pose_estimation/test/pose_classification_test.csv", number_of_keypoints=17)

# =============================================================================
# Calculate the class weights for the loss function
# =============================================================================
# Class 0: No Tackle
# Class 1: Low Tackle
# Class 2: High Tackle

# Location    Number of no tackle poses    Number of low tackle poses    Number of high tackle poses      Total number of clips
# All dirs:   1876                         33                            175                              2084
# Train:      1488                         27                            142                              1657
# Test:       201                          5                             15                               221
# Validation: 187                          1                             18                               206



# Location    Class 0 weight    Class 1 weight    Class 2 weight
# All dirs:   0.37              21.0              3.97
# Train:      0.37              20.5              3.89
# Test:       0.37              14.7              4.91
# Validation: 0.37              68.7              3.81

# def loss_function(output, target):
#     class_1_weight = 1.0
#     class_0_weight = 1.0
#     temp1 = torch.clamp(torch.log(  output), min=-100)
#     temp2 = torch.clamp(torch.log(1 - output), min=-100)

#     all_loss = -(class_1_weight * (target * temp1) + class_0_weight * ((1 - target) * temp2))
#     loss = torch.mean(all_loss)
#     return loss

def get_number_per_class(dir_path):
    p1_fields = np.array([[f"p1x{i}", f"p1y{i}"] for i in range(1, 18)]).flatten()
    p2_fields = np.array([[f"p2x{i}", f"p2y{i}"] for i in range(1, 18)]).flatten()
    field_names = ["video", "frame", *p1_fields, *p2_fields, "tackle_type"]
    number_of_no_tackles = [0, 0, 0]
    number_of_low_tackles = [0, 0, 0]
    number_of_high_tackles = [0, 0, 0]

    with open(os.path.join(dir_path, "test", "pose_classification_test.csv"), 'r') as f:
        csv_file = csv.DictReader(f, fieldnames=field_names)
        for row in csv_file:
            if row["tackle_type"] == "0":
                number_of_no_tackles[0] += 1
            elif row["tackle_type"] == "1":
                number_of_low_tackles[0] += 1
            elif row["tackle_type"] == "2":
                number_of_high_tackles[0] += 1
    with open(os.path.join(dir_path, "train", "pose_classification_train.csv"), 'r') as f:
        csv_file = csv.DictReader(f, fieldnames=field_names)
        for row in csv_file:
            if row["tackle_type"] == "0":
                number_of_no_tackles[1] += 1
            elif row["tackle_type"] == "1":
                number_of_low_tackles[1] += 1
            elif row["tackle_type"] == "2":
                number_of_high_tackles[1] += 1
    with open(os.path.join(dir_path, "valid", "pose_classification_valid.csv"), 'r') as f:
        csv_file = csv.DictReader(f, fieldnames=field_names)
        for row in csv_file:
            if row["tackle_type"] == "0":
                number_of_no_tackles[2] += 1
            elif row["tackle_type"] == "1":
                number_of_low_tackles[2] += 1
            elif row["tackle_type"] == "2":
                number_of_high_tackles[2] += 1
    return number_of_no_tackles, number_of_low_tackles, number_of_high_tackles


def train(model, train_loader, loss_function, optimiser, batch_size= 4):
    model.train()
    print("----------------- Starting Training -----------------")
    size = len(train_loader.dataset)
    current_complete = 0
    for batch, (name, frame, X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        X = X.to(dtype=torch.float32)
        X = torch.reshape(X, (X.shape[0], -1))

        optimiser.zero_grad()
        pred = model(X)
        loss = loss_function(pred, y)
        loss.backward()
        optimiser.step()
        current_complete += len(X)
        print(f"Training {current_complete}/{size}")

def validate(model, train_loader, loss_function, batch_size= 4):
    model.eval()
    print("----------------- Starting Validation -----------------")
    size = len(train_loader.dataset)
    computed_so_far = 0
    correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_id, (name, frame, X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            X = X.to(dtype=torch.float32)
            X = torch.reshape(X, (X.shape[0], -1))
            pred = model(X)    
            total_loss += loss_function(pred, y).item()
            all_preds = all_preds + pred.tolist()
            all_labels = all_labels + y.tolist()
            computed_so_far += len(X)
            print(f"[{computed_so_far:>5d}/{size:>5d}]")  
    

    all_preds = [np.argmax(pred) for pred in all_preds]
    print(f"Validation complete")
   
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, 
                         index = ["No Tackle", "Low Tackle", "High Tackle"],
                         columns = ["No Tackle", "Low Tackle", "High Tackle"])
    f_1 = f1_score(all_labels, all_preds, average=None)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Non Tackle F1: {f_1[0]} Low Tackle F1: {f_1[1]} High Tackle F1: {f_1[2]} weighted F1: {weighted_f1}")

    return (100*correct), total_loss/size,  weighted_f1, cm_df, f_1

def save_model(model, out_path):
    torch.save(model, out_path)

def split_data(data_path, train_size, validate_size, test_size):
    p1_fields = np.array([[f"p1x{i}", f"p1y{i}"] for i in range(1, 18)]).flatten()
    p2_fields = np.array([[f"p2x{i}", f"p2y{i}"] for i in range(1, 18)]).flatten()
    field_names = ["video", "frame", *p1_fields, *p2_fields, "tackle_type"]
    print("Field names created", field_names)
    print("Splitting data")
    # Create the files
    with open(os.path.join(data_path, "test", "pose_classification_test.csv"), 'w') as test_file:
        csv_file = csv.DictWriter(test_file, fieldnames=field_names)
        csv_file.writeheader()
    with open(os.path.join(data_path, "train", "pose_classification_train.csv"), 'w') as train_file:
        csv_file = csv.DictWriter(train_file, fieldnames=field_names)
        csv_file.writeheader()
    with open(os.path.join(data_path, "valid", "pose_classification_valid.csv"), 'w') as validate_file:
        csv_file = csv.DictWriter(validate_file, fieldnames=field_names)
        csv_file.writeheader()
    

    with open(os.path.join(data_path, "pose_classification.csv"), 'r') as f:
        csv_file = csv.DictReader(f, fieldnames=field_names)
        for row in csv_file:
            random_number = random.random()
            # print("Row", row, "Random number", random_number)
            if random_number < test_size:
                with open(os.path.join(data_path, "test", "pose_classification_test.csv"), 'a') as test_file:
                    csv_file = csv.DictWriter(test_file, fieldnames=field_names)
                    csv_file.writerow(row)
            elif random_number < test_size + validate_size:
                with open(os.path.join(data_path, "valid", "pose_classification_valid.csv"), 'a') as validate_file:
                    csv_file = csv.DictWriter(validate_file, fieldnames=field_names)
                    csv_file.writerow(row)
            else:
                with open(os.path.join(data_path, "train", "pose_classification_train.csv"), 'a') as train_file:
                    csv_file = csv.DictWriter(train_file, fieldnames=field_names)
                    csv_file.writerow(row)

    print("Data is split")
            
def plot_confusion_matrix(cm, path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(path)

def main(): 
    number_no_tackles, number_low_tackles, number_high_tackles = get_number_per_class("/dcs/large/u2102661/CS310/datasets/pose_estimation")
    print("Number of no tackles", number_no_tackles)
    print("Number of low tackles", number_low_tackles)
    print("Number of high tackles", number_high_tackles)
    print("total no tackles", sum(number_no_tackles),
          "total low tackles", sum(number_low_tackles),
          "total high tackles", sum(number_high_tackles))
    my_model = PoseClassifierModel(17).to(device)
    optimiser = torch.optim.Adam(my_model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.37, 20.5, 3.89]).to(device))
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    epochs = 100
    best_f1_score = 0
    best_epoch = 0
    number_since_best = 0
    patience = 50
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        train(my_model, train_loader, loss_function, optimiser, batch_size)
        accuracy, average_loss, f_1, cm_df, f1_array = validate(my_model, validate_loader, loss_function, batch_size)
        save_model(my_model, "/dcs/large/u2102661/CS310/models/pose_estimation/last.pt")
        print(f"Validation Error: \n Accuracy: {accuracy:>0.1f}%, Average loss: {average_loss:>8f} \n")
        plot_confusion_matrix(cm_df, "/dcs/large/u2102661/CS310/models/pose_estimation/latest_confusion_matrix.png")
        if f_1 > best_f1_score:
            best_f1_score = f_1
            best_epoch = e
            number_since_best = 0
            plot_confusion_matrix(cm_df, "/dcs/large/u2102661/CS310/models/pose_estimation/best_confusion_matrix.png")
            save_model(my_model, "/dcs/large/u2102661/CS310/models/pose_estimation/best.pt")
        else:
            number_since_best += 1
        if number_since_best > patience:
            print(f"Stopping early as no improvement in {patience} epochs")
            print(f"Best f1: {best_f1_score} at epoch {best_epoch}")
            break
    print("Done!")


if __name__ == "__main__":
    main()

