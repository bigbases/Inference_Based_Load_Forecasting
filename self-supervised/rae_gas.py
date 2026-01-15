import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "sm_86"
path = '/root/workspace/AMI/InferProj/self-supervised'

# library
import logging
import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
num_epochs = 1000
batch_size = 32
lr = 1e-3
hidden_size = 128 #논문은 40~90
sequence_length = 28*2 # 논문은 30~500
n_layers = 1 # encoder, decoder의 각 layer수

def to_var(t, **kwargs):
    # ToDo: check whether cuda Variable.
    t = t.to(device)
    return Variable(t, **kwargs)

def to_device(model):
    return model.to(device)

class LSTMEDModule(nn.Module):
    def __init__(self, n_features: int, hidden_size: int,
                  n_layers: tuple):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                                num_layers=self.n_layers).to(device)
        # self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                                num_layers=self.n_layers).to(device)
        # self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features).to(device)
        # self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return to_var(torch.Tensor(n_layers, batch_size, hidden_size).zero_()),to_var(torch.Tensor(n_layers, batch_size, hidden_size).zero_())

    def forward(self, ts_batch):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return output, enc_hidden[1][-1]
    
# dataset load
# 전기
elec_train = pd.read_csv(f'{path}/data/elec_clustering_train.csv', index_col=0)
elec_train.index = pd.to_datetime(elec_train.index)
elec_train = elec_train.resample(rule='12H').sum()
elec_test = pd.read_csv(f'{path}/data/elec_clustering_test.csv', index_col=0)
elec_test.index = pd.to_datetime(elec_test.index)
elec_test = elec_test.resample(rule='12H').sum()
elec_total = pd.concat([elec_train,elec_test],axis=1)
# 수도
water_train = pd.read_csv(f'{path}/data/water_clustering_train.csv', index_col=0)
water_train.index = pd.to_datetime(water_train.index)
water_train = water_train.resample(rule='12H').sum()
water_test = pd.read_csv(f'{path}/data/water_clustering_test.csv', index_col=0)
water_test.index = pd.to_datetime(water_test.index)
water_test = water_test.resample(rule='12H').sum()
water_total = pd.concat([water_train,water_test],axis=1)
#가스
gas_train = pd.read_csv(f'{path}/data/gas_clustering_train.csv', index_col=0)
gas_train.index = pd.to_datetime(gas_train.index)
gas_train = gas_train.resample(rule='12H').sum()
gas_test = pd.read_csv(f'{path}/data/gas_clustering_test.csv', index_col=0)
gas_test.index = pd.to_datetime(gas_test.index)
gas_test = gas_test.resample(rule='12H').sum()
gas_total = pd.concat([gas_train,gas_test],axis=1)
#온수
hotwater_train = pd.read_csv(f'{path}/data/hotwater_clustering_train.csv', index_col=0)
hotwater_train.index = pd.to_datetime(hotwater_train.index)
hotwater_train = hotwater_train.resample(rule='12H').sum()
hotwater_test = pd.read_csv(f'{path}/data/hotwater_clustering_test.csv', index_col=0)
hotwater_test.index = pd.to_datetime(hotwater_test.index)
hotwater_test = hotwater_test.resample(rule='12H').sum()
hotwater_total = pd.concat([hotwater_train,hotwater_test],axis=1)

region_A = elec_train.columns
region_B = elec_test.columns

train_index = 504*2
test_index = (504+28)*2
cluster_index = (504+28+28)*2

energy_A = elec_total
energy_B = water_total
energy_C = hotwater_total
energy_target = gas_total

train_A = energy_A[:train_index]
train_B = energy_B[:train_index]
train_C = energy_C[:train_index]
test_A = energy_A[train_index:test_index]
test_B = energy_B[train_index:test_index]
test_C = energy_C[train_index:test_index]
test_target = energy_target[train_index:test_index]
cluster_A = energy_A[test_index:cluster_index]
cluster_B = energy_B[test_index:cluster_index]
cluster_C = energy_C[test_index:cluster_index]
cluster_target = energy_target[test_index:cluster_index]

scaler_A = MinMaxScaler()
scaler_B = MinMaxScaler()
scaler_C = MinMaxScaler()
train_A = pd.DataFrame(scaler_A.fit_transform(train_A))
train_B = pd.DataFrame(scaler_B.fit_transform(train_B))
train_C = pd.DataFrame(scaler_C.fit_transform(train_C))
test_A = pd.DataFrame(scaler_A.transform(test_A))
test_B = pd.DataFrame(scaler_B.transform(test_B))
test_C = pd.DataFrame(scaler_C.transform(test_C))
cluster_A = pd.DataFrame(scaler_A.transform(cluster_A))
cluster_B = pd.DataFrame(scaler_B.transform(cluster_B))
cluster_C = pd.DataFrame(scaler_C.transform(cluster_C))

# 2-5
repr1_A = energy_A[train_index+14:test_index+14]
repr1_B = energy_B[train_index+14:test_index+14]
repr1_C = energy_C[train_index+14:test_index+14]
repr1_A = pd.DataFrame(scaler_A.transform(repr1_A))
repr1_B = pd.DataFrame(scaler_B.transform(repr1_B))
repr1_C = pd.DataFrame(scaler_C.transform(repr1_C))

# 3-6
repr2_A = energy_A[train_index+28:test_index+28]
repr2_B = energy_B[train_index+28:test_index+28]
repr2_C = energy_C[train_index+28:test_index+28]
repr2_A = pd.DataFrame(scaler_A.transform(repr2_A))
repr2_B = pd.DataFrame(scaler_B.transform(repr2_B))
repr2_C = pd.DataFrame(scaler_C.transform(repr2_C))

# 4-7
repr3_A = energy_A[train_index+42:test_index+42]
repr3_B = energy_B[train_index+42:test_index+42]
repr3_C = energy_C[train_index+42:test_index+42]
repr3_A = pd.DataFrame(scaler_A.transform(repr3_A))
repr3_B = pd.DataFrame(scaler_B.transform(repr3_B))
repr3_C = pd.DataFrame(scaler_C.transform(repr3_C))

data_train = []
for user in range(elec_total.shape[1]):
    for index in range(0,elec_total.shape[0]//(28*2)-2):
        window = []
        for period in range(index*28*2,(index+1)*28*2):
            window.append([train_A.iloc[period,user],train_B.iloc[period,user],train_C.iloc[period,user]])
        data_train.append(window)

data_test = []
for user in range(elec_total.shape[1]):
    for index in range(0,1):
        window = []
        for period in range(index*28*2,(index+1)*28*2):
            window.append([test_A.iloc[period,user],test_B.iloc[period,user],test_C.iloc[period,user]])
        data_test.append(window)

data_repr1 = []
for user in range(elec_total.shape[1]):
    for index in range(0,1):
        window = []
        for period in range(index*28*2,(index+1)*28*2):
            window.append([repr1_A.iloc[period,user],repr1_B.iloc[period,user],repr1_C.iloc[period,user]])
        data_repr1.append(window)

data_repr2 = []
for user in range(elec_total.shape[1]):
    for index in range(0,1):
        window = []
        for period in range(index*28*2,(index+1)*28*2):
            window.append([repr2_A.iloc[period,user],repr2_B.iloc[period,user],repr2_C.iloc[period,user]])
        data_repr2.append(window)

data_repr3 = []
for user in range(elec_total.shape[1]):
    for index in range(0,1):
        window = []
        for period in range(index*28*2,(index+1)*28*2):
            window.append([repr3_A.iloc[period,user],repr3_B.iloc[period,user],repr3_C.iloc[period,user]])
        data_repr3.append(window)

# dataset
data_train = np.array(data_train)
data_test = np.array(data_test)
data_repr1 = np.array(data_repr1)
data_repr2 = np.array(data_repr2)
data_repr3 = np.array(data_repr3)

class TimeSeriesDataset(Dataset):
    def __init__(self, X):
        self.X = X
        # self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]

ds_train = TimeSeriesDataset(data_train)
ds_test = TimeSeriesDataset(data_test)
ds_repr1 = TimeSeriesDataset(data_repr1)
ds_repr2 = TimeSeriesDataset(data_repr2)
ds_repr3 = TimeSeriesDataset(data_repr3)

train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=ds_test, batch_size=batch_size, drop_last=False, shuffle=False)
repr1_loader = DataLoader(dataset=ds_repr1, batch_size=batch_size, drop_last=False, shuffle=False)
repr2_loader = DataLoader(dataset=ds_repr2, batch_size=batch_size, drop_last=False, shuffle=False)
repr3_loader = DataLoader(dataset=ds_repr3, batch_size=batch_size, drop_last=False, shuffle=False)

num = int(sys.argv[1])
# num = 0
model = LSTMEDModule(data_train.shape[2], hidden_size, n_layers)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
val_list = []
val_output = []
val_true = []

model.train()
for epoch in trange(num_epochs):
    logging.debug(f'Epoch {epoch+1}/{num_epochs}.')
    for ts_batch in train_loader:
        ts_batch = ts_batch.to(device)
        output, _ = model(ts_batch)
        loss = criterion(output, ts_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        for ts_batch in test_loader:
            ts_batch = ts_batch.to(device)
            output, _ = model(ts_batch)
            val_loss += criterion(output, ts_batch.float()).item()
        val_list.append(val_loss)
        if epoch == num_epochs-1:
            val_true.append(ts_batch.cpu())
            val_output.append(output.cpu())

repr1_output = []
repr1_true = []
repr1_hidden = []
with torch.no_grad():
    for ts_batch in repr1_loader:
        ts_batch = ts_batch.to(device)
        output, hidden = model(ts_batch)
        repr1_hidden.append(hidden.cpu().numpy())
        repr1_true.append(ts_batch.cpu().numpy())
        repr1_output.append(output.cpu().numpy())

arr_hidden_1 = pd.DataFrame()
for i in range(33):
    for j in range(32):
        arr_hidden_1 = pd.concat([arr_hidden_1,pd.DataFrame(repr1_hidden[i][j]).transpose()],axis=0)
for j in range(24):
    arr_hidden_1 = pd.concat([arr_hidden_1,pd.DataFrame(repr1_hidden[33][j]).transpose()],axis=0)

repr2_output = []
repr2_true = []
repr2_hidden = []
with torch.no_grad():
    for ts_batch in repr2_loader:
        ts_batch = ts_batch.to(device)
        output, hidden = model(ts_batch)
        repr2_hidden.append(hidden.cpu().numpy())
        repr2_true.append(ts_batch.cpu().numpy())
        repr2_output.append(output.cpu().numpy())

arr_hidden_2 = pd.DataFrame()
for i in range(33):
    for j in range(32):
        arr_hidden_2 = pd.concat([arr_hidden_2,pd.DataFrame(repr2_hidden[i][j]).transpose()],axis=0)
for j in range(24):
    arr_hidden_2 = pd.concat([arr_hidden_2,pd.DataFrame(repr2_hidden[33][j]).transpose()],axis=0)

repr3_output = []
repr3_true = []
repr3_hidden = []
with torch.no_grad():
    for ts_batch in repr3_loader:
        ts_batch = ts_batch.to(device)
        output, hidden = model(ts_batch)
        repr3_hidden.append(hidden.cpu().numpy())
        repr3_true.append(ts_batch.cpu().numpy())
        repr3_output.append(output.cpu().numpy())

arr_hidden_3 = pd.DataFrame()
for i in range(33):
    for j in range(32):
        arr_hidden_3 = pd.concat([arr_hidden_3,pd.DataFrame(repr3_hidden[i][j]).transpose()],axis=0)
for j in range(24):
    arr_hidden_3 = pd.concat([arr_hidden_3,pd.DataFrame(repr3_hidden[33][j]).transpose()],axis=0)

arr_hidden_1.index = elec_total.columns
arr_hidden_1.to_csv('{}/result/rae_gas/arr_1_{}.csv'.format(path, num))
arr_hidden_2.index = elec_total.columns
arr_hidden_2.to_csv('{}/result/rae_gas/arr_2_{}.csv'.format(path, num))
arr_hidden_3.index = elec_total.columns
arr_hidden_3.to_csv('{}/result/rae_gas/arr_3_{}.csv'.format(path, num))
