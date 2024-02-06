import os 
import sys
import csv
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.io import read_video, VideoReader
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter


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
    is_test=False, #
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
        
def save_model(model, path):
    torch.save(model, path)

def main():
    torch.cuda.empty_cache()
    model = models.video.r3d_18(pretrained=True)
    my_model = ActionRecogniserModel(model, "r3d_18").to(device)
    my_model.parent_model.requires_grad_(False)
    best_f1 = -1
    best_f2 = -1
    number_since_best = 0
    best_epoch = 0

    # Print initial weights of last layer
    params = list(my_model.parameters())
    print(params[-2])
    print(params[-1])

    # Freeze the weights of the parent model
    # loss_fun = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(my_model.parameters(), lr = 0.001)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    validate_loader = DataLoader(validate_data, batch_size=16, shuffle=True)

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

def temp():
    model = models.video.r3d_18(pretrained=True)
    my_model = ActionRecogniserModel(model, "r3d_18").to(device)
    my_model.parent_model.requires_grad = False

    temp_data = ActionRecogniserDataset(
        annotation_file="/dcs/large/u2102661/CS310/datasets/activity_recogniser/temp/labels.csv",
        video_dir="/dcs/large/u2102661/CS310/datasets/activity_recogniser/temp",
        is_test=False
    )
    temp_loader = DataLoader(temp_data, batch_size=4, shusquffle=True)
    for batch_idx, (data, target) in enumerate(temp_loader):
        data, target = data.to(device), target.to(device)
        target = target.unsqueeze(1).to(dtype=torch.float32)

        output = my_model(data)

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(output)):
            print(f"output: {output[i]} target: {target[i]}")
            if output[i] >= 0.5 and target[i] == 1:
                TP += 1
            elif output[i] >= 0.5 and target[i] == 0:
                FP += 1
            elif output[i] < 0.5 and target[i] == 1:
                FN += 1
            elif output[i] < 0.5 and target[i] == 0:
                TN += 1
        
        f1 = 0
        
        print(f"TP: {TP} TN: {TN} FP: {FP} FN: {FN}")
        if (TP + FN) == 0:
            f1 = 0
        elif (TP + FP) == 0:
            f1 = 0
        else:
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            f1 = 2 * (precision * recall / (precision + recall))
        print(f"f1: {f1}")


        print("-------------------")
        loss_function(output, target)
        
def temp2():
    model = models.video.r3d_18(pretrained=True)
    my_model = ActionRecogniserModel(model, "r3d_18").to(device)
    my_model.parent_model.requires_grad_(False)

    print(my_model[0].weight)

    # target = torch.tensor([0,0,0,0,0,1,0,0,0,0]).to(dtype=torch.float32)
    # output = torch.tensor([0.0, 1.0, 0.00000001, 0.12, 0.08, 0.48, 0.24, 0.01, 0.22, 0.16]).to(dtype=torch.float32)

    # loss_function(output, target)
    # for i in range(20): 
    #     print(f"i: {i}")
    #     if i > 10:
    #         break
    
if __name__ == "__main__":
    # main()
    # temp()
    temp2()