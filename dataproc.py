# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class LSTMDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]

class GraphDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

class TransformerDataset(Dataset):
    def __init__(self, X, mask):
        self.X = X
        self.target = X
        self.mask = mask
        # self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.target[index], self.mask[index]

# +
class DataProc:
    def __init__(self, target):
        self.energy_list = ['elec', 'water', 'gas', 'hotwater']

        # Time Interval 1
        self.train_index = 504*2
        self.test_index = (504+28)*2
        self.cluster_index = (504+28+28)*2
        
        # Time Interval 2
        # self.train_index = 504*2 -300
        # self.test_index = (504+28)*2 -300
        # self.cluster_index = (504+28+28)*2 -300
        
        self.target = target
        self.path = '/root/workspace/AMI/InferProj'

    # 데이터 불러와 12시간 단위로 변환
    def load_data(self, name):
        train = pd.read_csv(f'{self.path}/data/{name}_clustering_train.csv', index_col=0)
        train.index = pd.to_datetime(train.index)
        train = train.resample(rule='12h').sum()
        test = pd.read_csv(f'{self.path}/data/{name}_clustering_test.csv', index_col=0)
        test.index = pd.to_datetime(test.index)
        test = test.resample(rule='12h').sum()
        total = pd.concat([train,test],axis=1)

        return train, test, total
    
    # 데이터 불러오기
    def load_original_data(self, name):
        train = pd.read_csv(f'{self.path}/data/{name}_clustering_train.csv', index_col=0)
        train.index = pd.to_datetime(train.index)
        test = pd.read_csv(f'{self.path}/data/{name}_clustering_test.csv', index_col=0)
        test.index = pd.to_datetime(test.index)
        total = pd.concat([train,test],axis=1)

        return train, test, total
    
    # 데이터 불러오기
    def load_customized_data(self, name, interval):
        train = pd.read_csv(f'{self.path}/data/{name}_clustering_train.csv', index_col=0)
        train.index = pd.to_datetime(train.index)
        train = train.resample(rule=interval).sum()
        test = pd.read_csv(f'{self.path}/data/{name}_clustering_test.csv', index_col=0)
        test.index = pd.to_datetime(test.index)
        test = test.resample(rule=interval).sum()
        total = pd.concat([train,test],axis=1)

        return train, test, total
    
    # 데이터 3분할: 학습용, 테스트용, 클러스터링용
    def total_train_test_cluster(self, data):
        total = data
        train = data[:self.train_index]
        test = data[self.train_index:self.test_index]
        cluster = data[self.test_index:self.cluster_index]

        return total, train, test, cluster
    
    # 데이터 전처리
    def preprocess(self):

        # 각 자원별 데이터 불러와 딕셔너리에 저장
        train, test, total = {}, {}, {}
        for i in self.energy_list:
            train[i], test[i], total[i] = self.load_data(i)
        
        self.elec_total = total["elec"]         # 전체 단지 전기 사용량
        self.region_A = train["elec"].columns   # A단지 목록
        self.region_B = test["elec"].columns    # B단지 목록

        # 유추 대상(=target) 사용량 저장
        self.target_total = total[self.target]
        self.target_cluster = total[self.target][self.test_index:self.cluster_index]
        
        # 유추 대상 제외한 사용량 저장
        except_target = self.energy_list.copy()
        except_target.remove(self.target)
        energy_total = [total[i] for i in except_target]
        self.original_total = energy_total

        # 데이터 정규화한 뒤 저장: 훈련용 데이터 기준으로 정규화
        # 코드 상으론 사용하지 않는 변수
        scalers = [MinMaxScaler()] * 3
        self.energy_total = []
        self.energy_train = []
        self.energy_test = []
        self.energy_cluster = []
        for i in range(3):
            total = energy_total[i]
            train = total[:self.train_index]

            scaler = scalers[i]
            train = scaler.fit_transform(train)
            total = scaler.transform(total)
            test = total[self.train_index:self.test_index]
            cluster = total[self.test_index:self.cluster_index]

            self.energy_total.append(total)
            self.energy_train.append(train)
            self.energy_test.append(test)
            self.energy_cluster.append(cluster)

    def noise_mask(self, X, masking_ratio=0.15, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
        """
        Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
        Args:
            X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
            masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
                feat_dim that will be masked on average
            lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
            mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
                should be masked concurrently ('concurrent')
            distribution: whether each mask sequence element is sampled independently at random, or whether
                sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
                masked squences of a desired mean length `lm`
            exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
        Returns:
            boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
        """
        if exclude_feats is not None:
            exclude_feats = set(exclude_feats)

        if distribution == 'geometric':  # stateful (Markov chain)
            if mode == 'separate':  # each variable (feature) is independent
                mask = np.ones(X.shape, dtype=bool)
                for m in range(X.shape[1]):  # feature dimension
                    if exclude_feats is None or m not in exclude_feats:
                        mask[:, m] = self.geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
            else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
                mask = np.tile(np.expand_dims(self.geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
        else:  # each position is independent Bernoulli with p = 1 - masking_ratio
            if mode == 'separate':
                mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                        p=(1 - masking_ratio, masking_ratio))
            else:
                mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                                p=(1 - masking_ratio, masking_ratio)), X.shape[1])

        return mask

    def geom_noise_mask_single(self, L, lm, masking_ratio):
        """
        Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
        proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
        Args:
            L: length of mask and sequence to be masked
            lm: average length of masking subsequences (streaks of 0s)
            masking_ratio: proportion of L to be masked
        Returns:
            (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
        """
        keep_mask = np.ones(L, dtype=bool)
        p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
        for i in range(L):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state

        return keep_mask

    # Transformer AutoEncoder 전용 데이터셋 생성
    def TAE_dataset(self):
        data_train = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,self.elec_total.shape[0]//(28*2)-2):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_train[0].iloc[period,user],self.energy_train[1].iloc[period,user],self.energy_train[2].iloc[period,user]])
                data_train.append(window)

        data_test = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,1):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_test[0].iloc[period,user],self.energy_test[1].iloc[period,user],self.energy_test[2].iloc[period,user]])
                data_test.append(window)
                
        data_cluster = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,1):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_cluster[0].iloc[period,user],self.energy_cluster[1].iloc[period,user],self.energy_cluster[2].iloc[period,user]])
                data_cluster.append(window)

        # dataset
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        data_cluster = np.array(data_cluster)

        # masking
        data_train_mask = []
        for i in range(data_train.shape[0]):
            data_train_mask.append(self.noise_mask(data_train[i]))
            
        data_test_mask = []
        for i in range(data_test.shape[0]):
            data_test_mask.append(self.noise_mask(data_test[i]))
            
        data_cluster_mask = []
        for i in range(data_cluster.shape[0]):
            data_cluster_mask.append(self.noise_mask(data_cluster[i]))
            
        data_train_mask = np.array(data_train_mask)
        data_test_mask = np.array(data_test_mask)
        data_train_mask = np.array(data_train_mask)

        self.ds_train = TransformerDataset(data_train, data_train_mask)
        self.ds_test = TransformerDataset(data_test, data_test_mask)
        self.ds_cluster = TransformerDataset(data_cluster, data_cluster_mask)
# -







