import os
import csv
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class PoseDataset(Dataset):
    """
    A custom dataset class for the pose classification task that inherits from the PyTorch Dataset 
    class. This class is used to load the pose data from a CSV file and return it in a format that
    can be used by the PyTorch DataLoader class.

    
    Class Variables:
        poses: A pandas dataframe that contains the pose data
        number_of_keypoints: The number of keypoints in each pose
        transform: A transformation to apply to the data

    Clas Labels:
        0: No Tackle
        1: Low Tackle
        2: High Tackle
    """
    def __init__(self, csv_file, number_of_keypoints, transform=None):
        self.poses = pd.read_csv(csv_file)
        # Each keypoint has an x and y coordinate and there are 2 players in each tackle
        self.number_of_keypoints = number_of_keypoints * 4
        self.transform = transform

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        """
            This function is used to get a single item from the dataset. It also applies any 
            transformations to the data that are specified in the transform parameter.
        
        Input:
            idx: The index of the item to get from the dataset
        
        Output:
            video_name: The name of the video that the pose is from
            frame_number: The frame number of the pose in the video
            pose_keypoints: The keypoints of the pose
            label: The label of the pose
        """
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
    """
        A simple multi-class classifier model that inherits from the PyTorch nn.Module class. 
        This model takes in a tensor of pose keypoints and outputs a tensor of probabilities
        for each class.

        Architecture:
            - Input layer with 68 neurons (17 keypoints for 2 players)
            - Fully connected layer with 128 neurons
            - Batch Normalisation
            - Fully connected layer with 64 neurons
            - Batch Normalisation
            - Fully connected layer with 32 neurons
            - Batch Normalisation
            - Fully connected layer with 3 neurons (output layer)
        
        Features:
            - Batch Normalisation
            - Dropout with a probability of 0.2
            - ReLU activation functions
            - Softmax activation function for the output layer

    """
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
        """
        This function completes forward propagation through the model on a given input tensor x.

        Input:
            x: The input tensor to the model

        Output:
            x: The output tensor from the model of the probabilities of each class
        """
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


# =============================================================================
# Calculate the class weights for the loss function
# =============================================================================
# Class 0: No Tackle
# Class 1: Low Tackle
# Class 2: High Tackle

# Location     Number of no tackle   Number of low tackle   Number of high tackle   Totals
# All dirs:    6542                  419                    278                     7239
# Train:       5234                  335                    222                     5791
# Test:        654                   42                     28                      724
# Validation:  654                   42                     28                      724

# Location      Class 0 weight    Class 1 weight    Class 2 weight
# All dirs:     0.369             5.759             8.680
# Train:        0.369             5.762             8.695
# Test:         0.369             5.746             8.620
# Validation:   0.369             5.746             8.620

def get_number_per_class(dir_path):
    """
    This function reads the pose_classification csv files and counts the number of each class
    in the test, train and validation sets.

    Input:
        dir_path: The path to the directory containing the train, valid and test directories
    
    Output:
        number_of_no_tackles: A list containing the number of no tackles in each set
        number_of_low_tackles: A list containing the number of low tackles in each set
        number_of_high_tackles: A list containing the number of high tackles in each set
    """
    p1_fields = np.array([[f"p1x{i}", f"p1y{i}"] for i in range(1, 18)]).flatten()
    p2_fields = np.array([[f"p2x{i}", f"p2y{i}"] for i in range(1, 18)]).flatten()
    field_names = ["video", "frame", *p1_fields, *p2_fields, "tackle_type"]
    number_of_no_tackles = [0, 0, 0]
    number_of_low_tackles = [0, 0, 0]
    number_of_high_tackles = [0, 0, 0]

    # An inner function to read and count each csv
    def read_csv(file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            csv_file = csv.DictReader(f, fieldnames=field_names)
            for row in csv_file:
                if row["tackle_type"] == "0":
                    number_of_no_tackles[1] += 1
                elif row["tackle_type"] == "1":
                    number_of_low_tackles[1] += 1
                elif row["tackle_type"] == "2":
                    number_of_high_tackles[1] += 1

    read_csv(os.path.join(dir_path, "test", "pose_classification_test.csv"))
    read_csv(os.path.join(dir_path, "train", "pose_classification_train.csv"))
    read_csv(os.path.join(dir_path, "valid", "pose_classification_valid.csv"))
    return number_of_no_tackles, number_of_low_tackles, number_of_high_tackles

def tackle_classification(model, player1_distances, player2_distances, local_device):
    """
    This function takes in the distances between the keypoints of one player and the estimated
    head position of the other player and classifies the tackle as no tackle, low tackle or high 
    tackle. This function is used in the pipeline to classify the poses.

    Input:
        model: The model to use for classification
        player1_distances: The distances between the keypoints of player 1 and the estimated head
                            position of player 2
        player2_distances: The distances between the keypoints of player 2 and the estimated head
                            position of player 1
        local_device: The device to use for the model
    
    Output:
        pred: The prediction of the model
    """
    model.eval()
    # Create the input tensor
    player1_distances = torch.tensor(player1_distances)
    player2_distances = torch.tensor(player2_distances)
    player1_distances = torch.flatten(player1_distances)
    player2_distances = torch.flatten(player2_distances)
    input_tensor = torch.cat((player1_distances, player2_distances))
    # Change the shape to be 1 x 68
    input_tensor = torch.reshape(input_tensor, (1, -1))
    input_tensor = input_tensor.to(local_device)
    input_tensor = input_tensor.to(dtype=torch.float32)
    # Get the prediction
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.argmax(pred)
        return pred

def train(model, train_loader, loss_function, optimiser):
    """
    This function trains the model on the training data.

    Input:
        model: The model to train
        train_loader: The DataLoader containing the training data
        loss_function: The loss function to use
        optimiser: The optimiser to use
    
    Output:
        A single epoch of training on the model
    """
    model.train()
    print("----------------- Starting Training -----------------")
    size = len(train_loader.dataset)
    current_complete = 0
    for _, (_, _, x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x = x.to(dtype=torch.float32)
        x = torch.reshape(x, (x.shape[0], -1))
        optimiser.zero_grad()
        pred = model(x)
        loss = loss_function(pred, y)
        loss.backward()
        optimiser.step()
        current_complete += len(x)
        print(f"Training {current_complete}/{size}")

def validate(model, valid_loader, loss_function):
    """
    This function validates the model on the validation data. A single call to this function
    will validate the model on the entire validation set.

    Input:
        model: The model to validate
        valid_loader: The DataLoader containing the validation data
        loss_function: The loss function to use
    
    Output:
        accuracy: The accuracy of the model on the validation set
        average_loss: The average loss of the model on the validation set
        weighted_f1: The weighted f1 score of the model on the validation set
        cm_df: The confusion matrix of the model on the validation set
        f_1: The f1 score of the model on the validation set
    """
    model.eval()
    print("----------------- Starting Validation -----------------")
    size = len(valid_loader.dataset)
    computed_so_far = 0
    correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    all_weights = []
    with torch.no_grad():
        for _, (_, _, x, y) in enumerate(valid_loader):
            x, y = x.to(device), y.to(device)
            x = x.to(dtype=torch.float32)
            x = torch.reshape(x, (x.shape[0], -1))
            pred = model(x)
            total_loss += loss_function(pred, y).item()
            all_preds = all_preds + pred.tolist()
            all_labels = all_labels + y.tolist()
            batch_weights = [0.34 if y == 0 else 5.759 if y == 1 else 8.580 for y in y.tolist()]
            all_weights = all_weights + batch_weights
            computed_so_far += len(x)
            print(f"[{computed_so_far:>5d}/{size:>5d}]")
    all_preds = [np.argmax(pred) for pred in all_preds]
    correct = sum([1 if all_preds[i] == all_labels[i] else 0 for i in range(len(all_preds))])
    print("Validation complete")
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    cm_df = pd.DataFrame(
        cm,
        index=["No Tackle", "Low Tackle", "High Tackle"],
        columns=["No Tackle", "Low Tackle", "High Tackle"]
    )
    f_1 = f1_score(all_labels, all_preds, average=None)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", sample_weight=all_weights)
    print(f"Non Tackle F1: {f_1[0]} Low Tackle F1: {f_1[1]} High Tackle F1: {f_1[2]} weighted F1: {weighted_f1}")
    return (correct / size * 100), total_loss/size,  weighted_f1, cm_df, f_1

def evaluate(model, test_loader, loss_function):
    """
    This function evaluates the model on the test data. A single call to this function will evaluate
    the model on the entire test set.

    Input:
        model: The model to evaluate
        test_loader: The DataLoader containing the test data
        loss_function: The loss function to use
    
    Output:
        accuracy: The accuracy of the model on the test set
        average_loss: The average loss of the model on the test set
        weighted_f1: The weighted f1 score of the model on the test set
        cm_df: The confusion matrix of the model on the test set
        f_1: The f1 score of the model on the test set
    """
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
        for _, (_, _, x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            x = x.to(dtype=torch.float32)
            x = torch.reshape(x, (x.shape[0], -1))
            pred = model(x)
            total_loss += loss_function(pred, y).item()
            all_preds = all_preds + pred.tolist()
            all_labels = all_labels + y.tolist()
            batch_weights = [0.34 if y == 0 else 5.759 if y == 1 else 8.580 for y in y.tolist()]
            all_weights = all_weights + batch_weights
            computed_so_far += len(x)
            print(f"[{computed_so_far:>5d}/{size:>5d}]")
    all_preds = [np.argmax(pred) for pred in all_preds]
    correct = sum([1 if all_preds[i] == all_labels[i] else 0 for i in range(len(all_preds))])
    print("Validation complete")
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    cm_df = pd.DataFrame(
        cm,
        index=["No Tackle", "Low Tackle", "High Tackle"],
        columns=["No Tackle", "Low Tackle", "High Tackle"]
    )
    f_1 = f1_score(all_labels, all_preds, average=None)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", sample_weight=all_weights)
    print(f"Non Tackle F1: {f_1[0]} Low Tackle F1: {f_1[1]} High Tackle F1: {f_1[2]} weighted F1: {weighted_f1}")
    return (correct / size * 100), total_loss/size,  weighted_f1, cm_df, f_1

def save_model(model, out_path):
    """
    This function saves a model to a given path.

    Input:
        model: The model to save
        out_path: The path to save the model to
    
    Output:
        The model is saved to the out_path
    """
    torch.save(model, out_path)

def split_data(data_path, train_size, validate_size, test_size):
    """
    This function splits the pose_classification.csv file into test, train and validation sets
    based on the given sizes.

    Input:
        data_path: The path to the directory containing the pose_classification.csv file
        train_size: The proportion of the data to use for training
        validate_size: The proportion of the data to use for validation
        test_size: The proportion of the data to use for testing
    
    Output:
        The data is split into test, train and validation sets
    """
    p1_fields = np.array([[f"p1x{i}", f"p1y{i}"] for i in range(1, 18)]).flatten()
    p2_fields = np.array([[f"p2x{i}", f"p2y{i}"] for i in range(1, 18)]).flatten()
    field_names = ["video", "frame", *p1_fields, *p2_fields, "tackle_type"]
    print("Field names created", field_names)
    print("Splitting data")
    # Create the files

    # Inner functions to create and append to the csv files make the code more readable
    def create_csv(file_path):
        with open(file_path, 'w', encoding="utf-8") as f:
            csv_file = csv.DictWriter(f, fieldnames=field_names)
            csv_file.writeheader()
    def append_csv(file_path, row):
        with open(file_path, 'a', encoding="utf-8") as f:
            csv_file = csv.DictWriter(f, fieldnames=field_names)
            csv_file.writerow(row)
    create_csv(os.path.join(data_path, "test", "pose_classification_test.csv"))
    create_csv(os.path.join(data_path, "train", "pose_classification_train.csv"))
    create_csv(os.path.join(data_path, "valid", "pose_classification_valid.csv"))

    with open(os.path.join(data_path, "pose_classification.csv"), 'r', encoding="utf-8") as f:
        csv_file = csv.DictReader(f, fieldnames=field_names)
        for row in csv_file:
            random_number = random.random()
            # print("Row", row, "Random number", random_number)
            if random_number < test_size:
                append_csv(os.path.join(data_path, "test", "pose_classification_test.csv"), row)
            elif random_number < test_size + validate_size:
                append_csv(os.path.join(data_path, "valid", "pose_classification_valid.csv"), row)
            else:
                append_csv(os.path.join(data_path, "train", "pose_classification_train.csv"), row)
    print("Data is split")

def plot_confusion_matrix(cm, path):
    """
    This function plots a confusion matrix and saves it to a given path.

    Input:
        cm: The confusion matrix to plot
        path: The path to save the confusion matrix to
    
    Output:
        The confusion matrix is saved to the given path
    """
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(path)

def main():
    """
    This script is used to train a pose classifier model on the pose_classification.csv file. The script has three 
    operations that can be performed: train, test and split. The train operation trains the model on the training
    data and saves the best model. The test operation tests the model on the test data and saves
    the confusion matrix. The split operation splits the pose_classification.csv file into test,
    train and validation sets based on the given proportions.

    Arguments:
        --data_path: The path to the directory containing the pose_classification.csv file
        --op: The operation to perform (train, test, split)
        --train_size: The proportion of the data to use for training
        --validate_size: The proportion of the data to use for validation
        --test_size: The proportion of the data to use for testing
        --model_path: The path to the model to use for testing
    """
    parser = argparse.ArgumentParser(description="Train a pose classifier model")
    parser.add_argument("--data_path", type=str, default="/dcs/large/u2102661/CS310/datasets/pose_estimation", help="The path to the directory containing the pose_classification.csv file")
    parser.add_argument("--op", type=str, default="train", help="The operation to perform (train, test, split)")
    parser.add_argument("--train_size", type=float, help="The proportion of the data to use for training")
    # 0.8
    parser.add_argument("--validate_size", type=float, help="The proportion of the data to use for validation")
    # 0.1
    parser.add_argument("--test_size", type=float, help="The proportion of the data to use for testing")
    # 0.1
    parser.add_argument("--model_path", type=str, help="The path to the model to use for testing")
    # /dcs/large/u2102661/CS310/models/pose_estimation/run_new_data_12/best.pt

    args = parser.parse_args()
    global device # A global variable to store the device to use for the model computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = PoseDataset(
        csv_file=f"{args.data_path}/train/pose_classification_train.csv",
        number_of_keypoints=17
    )
    validate_dataset = PoseDataset(
        csv_file=f"{args.data_path}/valid/pose_classification_valid.csv",
        number_of_keypoints=17
    )
    test_dataset = PoseDataset(
        csv_file=f"{args.data_path}/test/pose_classification_test.csv",
        number_of_keypoints=17
    )
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.34, 5.759, 8.580]).to(device))
    match args.op:
        case "split":
            split_data(args.data_path, args.train_size, args.validate_size, args.test_size)
            number_no_tackles, number_low_tackles, number_high_tackles = get_number_per_class(args.data_path)
            print("Number of no tackles", number_no_tackles)
            print("Number of low tackles", number_low_tackles)
            print("Number of high tackles", number_high_tackles)
            print("total no tackles", sum(number_no_tackles),
                  "total low tackles", sum(number_low_tackles),
                  "total high tackles", sum(number_high_tackles))
        case "test":
            my_model = torch.load(args.model_path, map_location=device)
            accuracy, average_loss, f1, cm, _ = evaluate(my_model, test_loader, loss_function)
            print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Average loss: {average_loss:>8f}\n")
            output_path = args.model_path.replace("best.pt", "test_confusion_matrix.png")
            plot_confusion_matrix(cm, output_path)
        case "train":
            my_model = PoseClassifierModel(17).to(device)
            optimiser = torch.optim.Adam(my_model.parameters(), lr=0.001, weight_decay=0.001)
            epochs = 100
            best_f1_score = 0
            best_epoch = 0
            number_since_best = 0
            patience = 40
            output_path = args.model_path.removesuffix("/best.pt")
            for e in range(epochs):
                print(f"Epoch {e + 1}\n-------------------------------")
                train(my_model, train_loader, loss_function, optimiser)
                accuracy, average_loss, f1, cm, _ = validate(my_model, validate_loader, loss_function)
                save_model(my_model, f"{output_path}/last.pt")
                print(f"F1: {f1}, Average loss: {average_loss:>8f} \n")
                plot_confusion_matrix(cm, f"{output_path}/latest_confusion_matrix.png")
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_epoch = e
                    number_since_best = 0
                    plot_confusion_matrix(cm, f"{output_path}/best_confusion_matrix.png")
                    save_model(my_model, f"{output_path}/best.pt")
                else:
                    number_since_best += 1
                if number_since_best > patience:
                    print(f"Stopping early as no improvement in {patience} epochs")
                    print(f"Best f1: {best_f1_score} at epoch {best_epoch}")
                    break
            print("Done!")

if __name__ == "__main__":
    main()
