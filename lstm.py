# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length, output_size):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.output_size = output_size

    def __len__(self):
        return len(self.X) - self.sequence_length - self.output_size + 1  # -1 은 마지막 인덱스 제외하고 나머지 인덱스

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.sequence_length]
        y = self.y[idx + self.sequence_length: idx + self.sequence_length + self.output_size]
        return x, y

# data_type: vec, poh
def load_data(data_type, scaler=False, sequence_length=48, output_size=24):
    data_path = "/workspace/AMI/MiraeHall/data"
    data = pd.read_csv(f"{data_path}/preprocessed_data_{data_type}.csv")
    data = data.set_index("tm", drop=True)
    data = data.values

    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    if scaler:
        target_scaler = MinMaxScaler()
        feature_scaler = MinMaxScaler()
        train_data[:, 0:1] = target_scaler.fit_transform(train_data[:, 0:1])
        test_data[:, 0:1] = target_scaler.transform(test_data[:, 0:1])
        train_data[:,1:] = feature_scaler.fit_transform(train_data[:,1:])
        test_data[:,1:] = feature_scaler.transform(test_data[:,1:])
    else:
        target_scaler = None

    trainset = TimeSeriesDataset(train_data, sequence_length, output_size)
    testset = TimeSeriesDataset(test_data, sequence_length, output_size)
    
    return trainset, testset, target_scaler

def generate_dataloaders(trainset, testset, batch_size=128):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    return trainloader, testloader

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    
    all_losses = []

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            all_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return model, all_losses

def test_model(model, testloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    concatenated_labels = np.concatenate(all_labels, axis=0)
    concatenated_predictions = np.concatenate(all_predictions, axis=0)

    # 오차 계산
    mape = calculate_mape(concatenated_labels, concatenated_predictions)
    mse = calculate_mse(concatenated_labels, concatenated_predictions)
    mae = calculate_mae(concatenated_labels, concatenated_predictions)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    return all_labels, all_predictions, (mape, mse, mae)

def calculate_mape(labels, predictions):
    mape = np.mean(np.abs((labels - predictions) / labels)) * 100
    return mape

def calculate_mse(labels, predictions):
    mse = np.mean(np.square(labels - predictions))
    return mse

def calculate_mae(labels, predictions):
    mae = np.mean(np.abs(labels - predictions))
    return mae

def plot_losses(all_losses, fig_name=None):
    import matplotlib.pyplot as plt
    plt.plot(all_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    if fig_name is not None:
        plt.savefig(f"/workspace/AMI/MiraeHall/exp_result/figure/loss_history/{fig_name}")
    plt.show()

def save_pickle(scaler, all_labels, all_predictions, all_losses, error, file_name):
    import pickle
    with open(f"/workspace/AMI/MiraeHall/exp_result/pickle/{file_name}.pkl", "wb") as f:
        pickle.dump((scaler, all_labels, all_predictions, all_losses, error), f)
