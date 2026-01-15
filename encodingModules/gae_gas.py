import pandas as pd
import numpy as np

import json
from datetime import datetime
import torch.nn as nn

from utils import *
from mtad_gat import MTAD_GAT

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

path = '/root/workspace/AMI/InferProj/data'
# dataset load
# 전기
elec_train = pd.read_csv(f'{path}/elec_clustering_train.csv', index_col=0)
elec_train.index = pd.to_datetime(elec_train.index)
elec_train = elec_train.resample(rule='12H').sum()
elec_test = pd.read_csv(f'{path}/elec_clustering_test.csv', index_col=0)
elec_test.index = pd.to_datetime(elec_test.index)
elec_test = elec_test.resample(rule='12H').sum()
elec_total = pd.concat([elec_train,elec_test],axis=1)
# 수도
water_train = pd.read_csv(f'{path}/water_clustering_train.csv', index_col=0)
water_train.index = pd.to_datetime(water_train.index)
water_train = water_train.resample(rule='12H').sum()
water_test = pd.read_csv(f'{path}/water_clustering_test.csv', index_col=0)
water_test.index = pd.to_datetime(water_test.index)
water_test = water_test.resample(rule='12H').sum()
water_total = pd.concat([water_train,water_test],axis=1)
#가스
gas_train = pd.read_csv(f'{path}/gas_clustering_train.csv', index_col=0)
gas_train.index = pd.to_datetime(gas_train.index)
gas_train = gas_train.resample(rule='12H').sum()
gas_test = pd.read_csv(f'{path}/gas_clustering_test.csv', index_col=0)
gas_test.index = pd.to_datetime(gas_test.index)
gas_test = gas_test.resample(rule='12H').sum()
gas_total = pd.concat([gas_train,gas_test],axis=1)
#온수
hotwater_train = pd.read_csv(f'{path}/hotwater_clustering_train.csv', index_col=0)
hotwater_train.index = pd.to_datetime(hotwater_train.index)
hotwater_train = hotwater_train.resample(rule='12H').sum()
hotwater_test = pd.read_csv(f'{path}/hotwater_clustering_test.csv', index_col=0)
hotwater_test.index = pd.to_datetime(hotwater_test.index)
hotwater_test = hotwater_test.resample(rule='12H').sum()
hotwater_total = pd.concat([hotwater_train,hotwater_test],axis=1)

region_A = elec_train.columns
region_B = elec_test.columns

train_index = 504*2
test_index = (504+28)*2
cluster_index = (504+28+28)*2

energy_A = elec_total.reset_index(drop=True)
energy_B = water_total.reset_index(drop=True)
energy_C = hotwater_total.reset_index(drop=True)
energy_target = gas_total.reset_index(drop=True)

train_A = energy_A[:train_index]
train_B = energy_B[:train_index]
train_C = energy_C[:train_index]
train_target = energy_target[:train_index+1]
test_A = energy_A[train_index:test_index+1]
test_B = energy_B[train_index:test_index+1]
test_C = energy_C[train_index:test_index+1]
test_target = energy_target[train_index:test_index+1]
cluster_A = energy_A[test_index:cluster_index+1]
cluster_B = energy_B[test_index:cluster_index+1]
cluster_C = energy_C[test_index:cluster_index+1]
cluster_target = energy_target[test_index:cluster_index+1]

scaler_A = MinMaxScaler()
scaler_B = MinMaxScaler()
scaler_C = MinMaxScaler()
scaler_target = MinMaxScaler()
train_A = pd.DataFrame(scaler_A.fit_transform(train_A))
train_B = pd.DataFrame(scaler_B.fit_transform(train_B))
train_C = pd.DataFrame(scaler_C.fit_transform(train_C))
train_target = pd.DataFrame(scaler_target.fit_transform(train_target))
test_A = pd.DataFrame(scaler_A.transform(test_A))
test_B = pd.DataFrame(scaler_B.transform(test_B))
test_C = pd.DataFrame(scaler_C.transform(test_C))
test_target = pd.DataFrame(scaler_target.transform(test_target))
cluster_A = pd.DataFrame(scaler_A.transform(cluster_A))
cluster_B = pd.DataFrame(scaler_B.transform(cluster_B))
cluster_C = pd.DataFrame(scaler_C.transform(cluster_C))
cluster_target = pd.DataFrame(scaler_target.transform(cluster_target))


X_train = []
y_train = []
for user in range(elec_total.shape[1]):
    for index in range(0,elec_total.shape[0]//(28*2)-3):
        X_window = []
        y_window = []
        for period in range(index*28*2,(index+1)*28*2):
            X_window.append([train_A.iloc[period,user],train_B.iloc[period,user],train_C.iloc[period,user]])
        y_window.append(train_A.iloc[period+1,user])    
        X_train.append(X_window)
        y_train.append(y_window)

X_test = []
y_test = []
for user in range(elec_total.shape[1]):
    for index in range(0,1):
        X_window = []
        y_window = []
        for period in range(index*28*2,(index+1)*28*2):
            X_window.append([test_A.iloc[period,user],test_B.iloc[period,user],test_C.iloc[period,user]])
        y_window.append(test_A.iloc[period+1,user])
        X_test.append(X_window)
        y_test.append(y_window)

X_cluster = []
y_cluster = []
for user in range(elec_total.shape[1]):
    for index in range(0,1):
        X_window = []
        y_window = []
        for period in range(index*28*2,(index+1)*28*2):
            X_window.append([cluster_A.iloc[period,user],cluster_B.iloc[period,user],cluster_C.iloc[period,user]])
        y_window.append(cluster_A.iloc[period+1,user])
        X_cluster.append(X_window)
        y_cluster.append(y_window)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_cluster = np.array(X_cluster)
y_cluster = np.array(y_cluster)

class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

train_ds = CustomDataset(X_train, y_train)
valid_ds = CustomDataset(X_test, y_test) # 이게 test set

train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, batch_size=32, val_split=0.2, shuffle=False, test_dataset=valid_ds
    )

class Trainer:
    """Trainer class for MTAD-GAT model.

    :param model: MTAD-GAT model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=32,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[2]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[2]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []

            for x, y in train_loader:
                x = x.to(self.device).float() # 32,56,1080?????
                y = y.to(self.device).float()
                self.optimizer.zero_grad()

                preds, recons, _, _ = self.model(x)
                # print('pp')
                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = forecast_loss + recon_loss

                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            # Evaluate on validation set
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss <= self.losses["val_total"][-1]:
                    self.save(f"model.pt")

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s)

        if val_loader is None:
            self.save(f"model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """

        self.model.eval()

        forecast_losses = []
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                x = x.to(self.device).float() # 32,56,1080?????
                y = y.to(self.device).float()

                preds, recons, _, _ = self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)
                # print('aa')
#                 print(y.shape, preds.shape)
#                 print(y.shape) # 32,1
#                 print(preds.shape) # 32,1
#                 print(recons.shape) # 32,28,3
                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_loss = np.sqrt((recon_losses ** 2).mean())

        total_loss = forecast_loss + recon_loss

        return forecast_loss, recon_loss, total_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)

id = datetime.now().strftime("%d%m%Y_%H%M%S")

window_size = 28*2
n_epochs = 500 #30
batch_size = 32
init_lr = 1e-3
print_every = 1
log_tensorboard = True

n_features=X_train.shape[2]
window_size=window_size
out_dim=X_train.shape[2]
kernel_size=3
use_gatv2=True
feat_gat_embed_dim=None
time_gat_embed_dim=None
gru_n_layers=1
gru_hid_dim=256
forecast_n_layers=2
forecast_hid_dim=512
recon_n_layers=2
recon_hid_dim=128
dropout=0.2
alpha=0.2

for num in range(100,105):

    model = MTAD_GAT(
        n_features=n_features,
        window_size=window_size,
        out_dim=out_dim,
        kernel_size=kernel_size,
        use_gatv2=True,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        gru_n_layers=gru_n_layers,
        gru_hid_dim=gru_hid_dim,
        forecast_n_layers=forecast_n_layers,
        forecast_hid_dim=forecast_hid_dim,
        recon_n_layers=recon_n_layers,
        recon_hid_dim=recon_hid_dim,
        dropout=0.2,
        alpha=0.2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        window_size=window_size,
        n_features=n_features,
        n_epochs=n_epochs,
        batch_size=batch_size,
        init_lr=init_lr,
        forecast_criterion=forecast_criterion,
        recon_criterion=recon_criterion,
        use_cuda=True,
        print_every=1,
        log_dir="/logs",
        dload="/result"
    )

    trainer.fit(train_loader, val_loader)

    # hidden representations생성
    test_ds = CustomDataset(X_test, y_test) # 이게 test set
    cluster_ds = CustomDataset(X_cluster, y_cluster) # 이게 cluster set
    testset_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
    clusterset_loader = torch.utils.data.DataLoader(cluster_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forecast_criterion=nn.MSELoss()
    recon_criterion=nn.MSELoss()

    hidden_test = []
    hidden_cluster = []
    h_r_test = []
    h_r_cluster = []
    with torch.no_grad():
        for x, y in testset_loader:
            # x_true.append(np.array(x))
            # y_true.append(np.array(y))
            x = x.to(device).float()
            y = y.to(device).float()

            preds, recons, h, h_r = model(x)

            if preds.ndim == 3:
                preds = preds.squeeze(1)
            if y.ndim == 3:
                y = y.squeeze(1)

            # forecast.append(preds.cpu())
            # recon.append(recons.cpu())
            hidden_test.append(h.cpu())
            h_r_test.append(h_r.cpu())

    with torch.no_grad():
        for x, y in clusterset_loader:
            # x_true.append(np.array(x))
            # y_true.append(np.array(y))
            x = x.to(device).float()
            y = y.to(device).float()

            preds, recons, h, h_r = model(x)

            if preds.ndim == 3:
                preds = preds.squeeze(1)
            if y.ndim == 3:
                y = y.squeeze(1)

            # forecast.append(preds.cpu())
            # recon.append(recons.cpu())
            hidden_cluster.append(h.cpu())
            h_r_cluster.append(h_r.cpu())

    hidden_test_ = [np.array(i.cpu().squeeze()) for j in hidden_test for i in j]
    hidden_test_ = np.array(hidden_test_)
    hidden_test_ = pd.DataFrame(hidden_test_, index = elec_total.columns)
    hidden_cluster_ = [np.array(i.cpu().squeeze()) for j in hidden_cluster for i in j]
    hidden_cluster_ = np.array(hidden_cluster_)
    hidden_cluster_ = pd.DataFrame(hidden_cluster_, index = elec_total.columns)
    h_r_test_ = [np.array(i.cpu().squeeze()) for j in h_r_test for i in j]
    h_r_test_ = np.array(h_r_test_).reshape(1080,-1)
    h_r_test_ = pd.DataFrame(h_r_test_, index = elec_total.columns)
    h_r_cluster_ = [np.array(i.cpu().squeeze()) for j in h_r_cluster for i in j]
    h_r_cluster_ = np.array(h_r_cluster_).reshape(1080,-1)
    h_r_cluster_ = pd.DataFrame(h_r_cluster_, index = elec_total.columns)

    hidden_test_.to_csv('arr_hidden_test_12H_{}.csv'.format(num))
    hidden_cluster_.to_csv('arr_hidden_cluster_12H_{}.csv'.format(num))
    h_r_test_.to_csv('h_r_test_12H_{}.csv'.format(num))
    h_r_cluster_.to_csv('h_r_cluster_12H_{}.csv'.format(num))
    torch.save(model, 'graph_model_hidden_500_12H_{}.pt'.format(num))