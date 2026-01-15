import pandas as pd
import numpy as np
import sys
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error

# 경로 추가
path = '/workspace/AMI/jspark'
sys.path.append(path)

from dataproc import DataProc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 클래스
class LSTMDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]

# 데이터 처리 모델
class LSTMDataProc(DataProc):
    # LSTM AutoEncoder 전용 데이터셋 생성
    def RAE_dataset(self):
        # 전처리
        self.preprocess()
        
        data_train = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,self.elec_total.shape[0]//(28*2)-2):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_train[0][period,user],self.energy_train[1][period,user],self.energy_train[2][period,user]])
                data_train.append(window)

        data_test = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,1):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_test[0][period,user],self.energy_test[1][period,user],self.energy_test[2][period,user]])
                data_test.append(window)
                
        data_cluster = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,1):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_cluster[0][period,user],self.energy_cluster[1][period,user],self.energy_cluster[2][period,user]])
                data_cluster.append(window)

        # dataset
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        data_cluster = np.array(data_cluster)

        self.data_train = data_train
        self.data_test = data_test
        self.data_cluster = data_cluster

        # self.ds_train = LSTMDataset(torch.tensor(data_train))
        # self.ds_test = LSTMDataset(torch.tensor(data_test))
        # self.ds_cluster = LSTMDataset(torch.tensor(data_cluster))

        self.ds_train = LSTMDataset(data_train)
        self.ds_test = LSTMDataset(data_test)
        self.ds_cluster = LSTMDataset(data_cluster)

def to_var(t, **kwargs):
    # ToDo: check whether cuda Variable.
    t = t.to(device)
    return Variable(t, **kwargs)

# LSTM 모듈
class LSTMEDModule(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int):
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
            var1 = to_var(torch.Tensor(self.n_layers, batch_size, self.hidden_size).zero_())
            var2 = to_var(torch.Tensor(self.n_layers, batch_size, self.hidden_size).zero_())
            return var1, var2

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

# LSTM 모듈 사용해 표현 생성 및 저장
class LSTMAE:
    def __init__(self, target,
                 num_epochs=1000, batch_size=32, learning_rate=1e-3,
                 hidden_size=128, sequence_length=28*2, n_layers=1):

        # 데이터 처리용 클래스
        self.dp = LSTMDataProc(target)
        self.dp.RAE_dataset() # 전용 데이터셋 생성
        self.train_loader = DataLoader(dataset=self.dp.ds_train, batch_size=batch_size, drop_last=True, shuffle=True)
        self.test_loader = DataLoader(dataset=self.dp.ds_test, batch_size=batch_size, drop_last=False, shuffle=False)
        self.cluster_loader = DataLoader(dataset=self.dp.ds_cluster, batch_size=batch_size, drop_last=False, shuffle=False)

        #  Hyper-parameters
        self.target = target
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.n_layers = n_layers # encoder, decoder의 각 layer수

    def train(self):
        # 모델 정의
        model = LSTMEDModule(self.dp.data_train.shape[2], self.hidden_size, self.n_layers)
        model = model.to(device)

        # 최적화(Adam) 및 손실 함수(MSE) 설정
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        val_list = []
        val_output = []
        val_true = []

        # 학습
        for epoch in tqdm(range(self.num_epochs)):
            model.train()
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in self.train_loader:
                ts_batch = ts_batch.to(device)
                output, _ = model(ts_batch)
                loss = criterion(output, ts_batch.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 검증
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for ts_batch in self.test_loader:
                    ts_batch = ts_batch.to(device)
                    output, _ = model(ts_batch)
                    val_loss += criterion(output, ts_batch.float()).item()
                val_list.append(val_loss)
                if epoch == self.num_epochs-1:
                    val_true.append(ts_batch.cpu())
                    val_output.append(output.cpu())

        self.model = model
    
    # 은닉층 생성
    def predict(self, num):
        cluster_true = []
        cluster_hidden = []
        cluster_output = []
        with torch.no_grad():
            for ts_batch in self.cluster_loader:
                ts_batch = ts_batch.to(device)
                output, hidden = self.model(ts_batch)
                cluster_true.append(ts_batch.cpu().numpy())
                cluster_hidden.append(hidden.cpu().numpy())
                cluster_output.append(output.cpu().numpy())

        arr_true = pd.DataFrame()
        arr_hidden_cluster = pd.DataFrame()
        arr_output = pd.DataFrame()
        for i in range(33):
            for j in range(32):
                arr_hidden_cluster = pd.concat([arr_hidden_cluster,pd.DataFrame(cluster_hidden[i][j]).transpose()],axis=0)
                for k in range(28*2):
                    arr_true = pd.concat([arr_true,pd.DataFrame(cluster_true[i][j][k]).transpose()],axis=0)
                    arr_output = pd.concat([arr_output,pd.DataFrame(cluster_output[i][j][k]).transpose()],axis=0)
        for j in range(24):
            arr_hidden_cluster = pd.concat([arr_hidden_cluster,pd.DataFrame(cluster_hidden[33][j]).transpose()],axis=0)
            for k in range(28*2):
                arr_true = pd.concat([arr_true,pd.DataFrame(cluster_true[33][j][k]).transpose()],axis=0)
                arr_output = pd.concat([arr_output,pd.DataFrame(cluster_output[33][j][k]).transpose()],axis=0)
        arr_hidden_cluster.index = self.dp.elec_total.columns
        print(mean_absolute_error(arr_true, arr_output))    # LSTM 모델 오차 출력

        test_true = []
        test_hidden = []
        test_output = []
        with torch.no_grad():
            for ts_batch in self.test_loader:
                ts_batch = ts_batch.to(device)
                output, hidden = self.model(ts_batch)
                test_hidden.append(hidden.cpu().numpy())
                test_true.append(ts_batch.cpu().numpy())
                test_output.append(output.cpu().numpy())

        arr_hidden_test = pd.DataFrame()
        for i in range(33):
            for j in range(32):
                arr_hidden_test = pd.concat([arr_hidden_test,pd.DataFrame(test_hidden[i][j]).transpose()],axis=0)
        for j in range(24):
            arr_hidden_test = pd.concat([arr_hidden_test,pd.DataFrame(test_hidden[33][j]).transpose()],axis=0)
        arr_hidden_test.index = self.dp.elec_total.columns
            
        # 은닉층 및 모델 저장
        arr_hidden_cluster.to_csv(f'{path}/hidden_layers/lstm_hidden_cluster_{self.target}_12H_{num}.csv')
        arr_hidden_test.to_csv(f'{path}/hidden_layers/lstm_hidden_test_{self.target}_12H_{num}.csv')
        torch.save(self.model, f'{path}/models/LSTMAE_12H_{self.target}_{num}.pt')
        
