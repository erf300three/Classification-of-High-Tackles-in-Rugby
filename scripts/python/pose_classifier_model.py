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
            pose_keypoints.append([float(self.poses.iloc[idx, 2 + i])])
        pose_keypoints = torch.tensor(pose_keypoints)
        label = float(self.poses.iloc[idx, 2 + self.number_of_keypoints])
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            pose_keypoints = self.transform(pose_keypoints)
        
        return video_name, frame_number, pose_keypoints, label
        
        

# Create the custom classifying model
# This is a simple multi-class classifier with 4 fully connected layers the number 
class PoseClassifierModel(nn.Module):
    def __init__(self, number_of_key_points):
        super(PoseClassifierModel, self).__init__()
        self.fc1 = nn.Linear(number_of_key_points * 4, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 3)
        self.softmax = nn.Softmax()
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
        x = self.softmax(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# training_transforms = transforms.Compose([
#     lambda x: nn.functional.dropout(x, p=0.05)
# ])

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
# All dirs:   6542                         419                           278                              7239
# Train:      5234                         335                           222                              5791
# Test:       654                          42                            28                               724
# Validation: 654                          42                            28                               724



# Location    Class 0 weight    Class 1 weight    Class 2 weight
# All dirs:   0.369             5.759             8.680
# Train:      0.369             5.762             8.695
# Test:       0.369             5.746             8.620
# Validation: 0.369             5.746             8.620

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

# This is the function that we will use in the pipeline to classify the poses as no tackle, low tackle or high tackle
def tackle_classification(model, player1_distances, player2_distances, device):
    model.eval()
    # Create the input tensor
    player1_distances = torch.tensor(player1_distances)
    player2_distances = torch.tensor(player2_distances)
    player1_distances = torch.flatten(player1_distances)
    player2_distances = torch.flatten(player2_distances)
    input_tensor = torch.cat((player1_distances, player2_distances))
    # Change the shape to be 1 x 68
    input_tensor = torch.reshape(input_tensor, (1, -1))
    input_tensor = input_tensor.to(device)
    input_tensor = input_tensor.to(dtype=torch.float32)
    # Get the prediction
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.argmax(pred)
        return pred


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
    all_weights = []
    with torch.no_grad():
        for batch_id, (name, frame, X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            X = X.to(dtype=torch.float32)
            X = torch.reshape(X, (X.shape[0], -1))
            pred = model(X)    
            total_loss += loss_function(pred, y).item()
            all_preds = all_preds + pred.tolist()
            all_labels = all_labels + y.tolist()
            all_weights = all_weights + [0.34 if y == 0 else 5.759 if y == 1 else 8.580 for y in y.tolist()]
            computed_so_far += len(X)
            print(f"[{computed_so_far:>5d}/{size:>5d}]")  
    

    all_preds = [np.argmax(pred) for pred in all_preds]
    correct = sum([1 if all_preds[i] == all_labels[i] else 0 for i in range(len(all_preds))])
    print(f"Validation complete")
   
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    cm_df = pd.DataFrame(cm, 
                         index = ["No Tackle", "Low Tackle", "High Tackle"],
                         columns = ["No Tackle", "Low Tackle", "High Tackle"])
    f_1 = f1_score(all_labels, all_preds, average=None)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", sample_weight=all_weights)

    print(f"Non Tackle F1: {f_1[0]} Low Tackle F1: {f_1[1]} High Tackle F1: {f_1[2]} weighted F1: {weighted_f1}")

    return (correct / size * 100), total_loss/size,  weighted_f1, cm_df, f_1

def evaluate(model, test_loader, loss_function, batch_size=4):
    model.eval()
    print("----------------- Starting Evaluation -----------------")
    size = len(test_loader.dataset)
    computed_so_far = 0
    correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    all_weights = []
    with torch.no_grad():
        for batch_id, (name, frame, X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            X = X.to(dtype=torch.float32)
            X = torch.reshape(X, (X.shape[0], -1))
            pred = model(X)    
            total_loss += loss_function(pred, y).item()
            all_preds = all_preds + pred.tolist()
            all_labels = all_labels + y.tolist()
            all_weights = all_weights + [0.34 if y == 0 else 5.759 if y == 1 else 8.580 for y in y.tolist()]
            computed_so_far += len(X)
            print(f"[{computed_so_far:>5d}/{size:>5d}]")  
    

    all_preds = [np.argmax(pred) for pred in all_preds]
    correct = sum([1 if all_preds[i] == all_labels[i] else 0 for i in range(len(all_preds))])
    print(f"Validation complete")
   
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    cm_df = pd.DataFrame(cm, 
                         index = ["No Tackle", "Low Tackle", "High Tackle"],
                         columns = ["No Tackle", "Low Tackle", "High Tackle"])
    f_1 = f1_score(all_labels, all_preds, average=None)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", sample_weight=all_weights)

    print(f"Non Tackle F1: {f_1[0]} Low Tackle F1: {f_1[1]} High Tackle F1: {f_1[2]} weighted F1: {weighted_f1}")

    return (correct / size * 100), total_loss/size,  weighted_f1, cm_df, f_1

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

def main(is_test=False): 
    # split_data("/dcs/large/u2102661/CS310/datasets/pose_estimation", 0.8, 0.1, 0.1)
    # In the order of test, train, validation
    number_no_tackles, number_low_tackles, number_high_tackles = get_number_per_class("/dcs/large/u2102661/CS310/datasets/pose_estimation")
    print("Number of no tackles", number_no_tackles)
    print("Number of low tackles", number_low_tackles)
    print("Number of high tackles", number_high_tackles)
    print("total no tackles", sum(number_no_tackles),
          "total low tackles", sum(number_low_tackles),
          "total high tackles", sum(number_high_tackles))
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.34, 5.759, 8.580]).to(device))
    if is_test:
        my_model = torch.load("/dcs/large/u2102661/CS310/models/pose_estimation/best.pt", map_location=device)
        accuracy, average_loss, f_1, cm_df, f1_array = evaluate(my_model, test_loader, loss_function, batch_size)
        print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Average loss: {average_loss:>8f} \n")
        plot_confusion_matrix(cm_df, "/dcs/large/u2102661/CS310/model_evaluation/pose_classification/test_confusion_matrix.png")
        return

    my_model = PoseClassifierModel(17).to(device)
    optimiser = torch.optim.Adam(my_model.parameters(), lr=0.001, weight_decay=0.001)
    epochs = 100
    best_f1_score = 0
    best_epoch = 0
    number_since_best = 0
    patience = 40
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
    main(is_test=False)
    main(is_test=True)

