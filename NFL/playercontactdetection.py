import os
import numpy as np
import pandas as pd
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import cv2
import time
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# for test.csv
def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols]
    output_cols += [c+"_2" for c in use_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
        
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    output_cols += ["G_flug"]
    return df_combo, output_cols


use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa'
]

# all data are on Kaggle

train_labels = pd.read_csv("/kaggle/input/nfl-player-contact-detection/train_labels.csv")
train_tracking = pd.read_csv("/kaggle/input/nfl-player-contact-detection/train_player_tracking.csv")
train_video_metadata = pd.read_csv("/kaggle/input/nfl-player-contact-detection/train_video_metadata.csv")
labels = expand_contact_id(pd.read_csv("/kaggle/input/nfl-player-contact-detection/sample_submission.csv"))
test_tracking = pd.read_csv("/kaggle/input/nfl-player-contact-detection/test_player_tracking.csv")

#train_labels
train_data, feature_cols = create_features(train_labels, train_tracking, use_cols=use_cols)

test, feature_cols = create_features(labels, test_tracking, use_cols=use_cols)

filtered_cols = feature_cols + ['contact']

train_filtered = train_data[filtered_cols].query('distance < 2.5')
test_filtered = test[filtered_cols].query('distance < 2.5')

train_features = torch.tensor(train_filtered[feature_cols].sample(n=20000, random_state=15).values.astype(np.float32))
train_labels = torch.tensor(train_filtered['contact'].sample(n=20000, random_state=15).values.astype(np.float32))
test_features = torch.tensor(test_filtered[feature_cols].values.astype(np.float32))
test_labels = torch.tensor(test_filtered['contact'].values.astype(np.float32))

train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset)

test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset)

#ANN
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(18, 64)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feature):
        output = self.fc1(feature)
        output = self.elu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        #output = torch.round(output)
        return output
    
model = ANNModel()

error = nn.BCELoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_iters = 20
num_epochs = n_iters

#training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        train = torch.autograd.Variable(features)
        labels = torch.autograd.Variable(labels).unsqueeze(1)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        loss = error(outputs, labels)
        #print(outputs, labels, loss)
        loss.backward()
        
        # Update parameters
        optimizer.step()
        '''for parameter in model.parameters():
            print(parameter.data)'''
        count += 1
        
        if count % 50 == 0:
            #print(outputs, labels, loss)
            #for parameter in model.parameters():
                #print('data: ', parameter.data)
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for features, labels in test_loader:
                test = torch.autograd.Variable(features)
                labels = torch.autograd.Variable(labels).unsqueeze(1)
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = outputs
                #print('outputs: ', predicted.data, 'labels: ', labels)
                
                # Total number of labels
                total += len(labels)

                # Total correct predictions
                correct += (torch.round(predicted.data) == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 1000 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
            
# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of iteration")
plt.show()
