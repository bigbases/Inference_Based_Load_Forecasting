from dataproc import DataProc

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from random import uniform

from sklearn.ensemble import RandomForestRegressor
from sklearn_extra.cluster import KMedoids, CLARA, CommonNNClustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xgboost as xgb

# from tslearn.clustering import TimeSeriesKMeans, KShape
from dtaidistance import ed, dtw, dtw_ndim
from scipy.cluster.hierarchy import linkage, fcluster

class Predict:
    def __init__(self, infer_target, pred_target):
        self.infer_target = infer_target
        self.pred_target = pred_target
        total_list = ['elec','gas','water','hotwater','inferred','random']
        self.exp_list = []
        self.exp_list.append([i for i in total_list if i not in ['random', infer_target]])              # 4종 유추
        self.exp_list.append([i for i in total_list if i not in ['inferred', infer_target]])            # 4종 랜덤
        self.exp_list.append([i for i in total_list if i not in ['inferred', 'random']])                # 4종 실제
        self.exp_list.append([i for i in total_list if i not in ['inferred', 'random', infer_target]])  # 3종

        dp = DataProc(infer_target)
        dp.preprocess()

        self.region_A = dp.region_A
        self.region_B = dp.region_B
        total_region = dp.elec_total.columns

        self.original_total = dp.original_total
        self.energy_total = dp.energy_total
        self.target_total = dp.target_total
        self.target_cluster = dp.target_cluster

        daterange = dp.elec_total.index
        self.energy_A = pd.DataFrame(dp.original_total[0], index=daterange, columns=total_region)
        self.energy_B = pd.DataFrame(dp.original_total[1], index=daterange, columns=total_region)
        self.energy_C = pd.DataFrame(dp.original_total[2], index=daterange, columns=total_region)

        self.train_index = dp.train_index
        self.test_index = dp.test_index

    def multi_DTW(self, start, end):
        n_houses = e1.shape[1]
        e1 = self.energy_total[0][start:end].T.reshape(n_houses,-1,1)
        e2 = self.energy_total[1][start:end].T.reshape(n_houses,-1,1)
        e3 = self.energy_total[2][start:end].T.reshape(n_houses,-1,1)

        e1 = MinMaxScaler().fit_transform(e1)
        e2 = MinMaxScaler().fit_transform(e2)
        e3 = MinMaxScaler().fit_transform(e3)

        data = np.c_[e1, e2, e3]
        dist_matrix = [[dtw_ndim.distance(data[:,i],data[:,j]) for i in range(n_houses)] for j in range(n_houses)]

        return dist_matrix

    # data
    def FDM(self, start, end, dist_model='euclidean'):
        norm_list = []
        for i in self.energy_total:
            i = i[start:end]
            # Replace NaN values with 0 (or any other desired value)
            i[np.isnan(i)] = 0

            n_houses = i.shape[1]
            # DTW
            if dist_model == 'dtw':
                ds = dtw.distance_matrix_fast(i.T)
                scaled_ds = MinMaxScaler().fit_transform(ds)
                norm = pd.DataFrame(scaled_ds)
                norm_list.append(norm)
            # Euclidean
            elif dist_model == 'euclidean':
                fdm = i
                rows = [[ed.distance(fdm[:,j],fdm[:,k]) for j in range(n_houses)] for k in range(n_houses)]
                dist = np.array(rows)
                dist = pd.DataFrame(dist)
                norm = dist/dist.mean().mean()
                norm_list.append(norm)

        fdm = norm_list[0] + norm_list[1] + norm_list[2]
        
        return fdm

    def clustering(self, fdm, start, end):
        n_clusters=5
        kmeans = KMedoids(n_clusters=n_clusters, random_state=42, metric='precomputed').fit(fdm)
        labels=kmeans.labels_
        #각 클러스터의 중심으로 water 사용량 할당
        target_test_result = pd.DataFrame(columns=self.region_B,index=self.target_total[start:end].index)
        for j in range(n_clusters):
            for k in list(set(self.region_B) & set(self.target_total.columns[labels==j])):
                target_test_result[k] = self.target_total[list(set(self.region_A) & set(self.target_cluster.columns[labels==j]))][start:end].mean(axis=1)

        #클러스터 구성요소가 B단지 1가구인 클러스터는 A단지 평균으로 할당
        for k in set(target_test_result.columns) - set(target_test_result.dropna(axis=1).columns):
            target_test_result[k] = self.target_total[self.region_A][start:end].mean(axis=1)

        return target_test_result
    
    def clustering_new(self, start, end, mode='kmeans'):

        # 단순 concatenate
        n_clusters=5
        for n, i in enumerate(self.energy_total):
            energy = i[start:end]
            energy = MinMaxScaler().fit_transform(energy)
            if n == 0:
                data = energy
            else:
                data = np.r_[data, energy]
        data = data.T

        if mode == 'kmeans':
            labels = KMeans(n_clusters=n_clusters, random_state=42).fit(data).labels_
        elif mode == 'agglo':
            labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(data).labels_
        # elif mode == 'tskmeans':
        #     labels = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42, n_jobs=-1).fit(data).labels_
        # elif mode == 'kshape':
        #     labels = KShape(n_clusters=n_clusters, random_state=42).fit(data).labels_
        elif mode == 'kmedoids':
            labels = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean').fit(data).labels_
        elif mode == 'hierarchical':
            Z = linkage(data, 'ward')
            labels = fcluster(Z, n_clusters, criterion='maxclust')

        #각 클러스터의 중심으로 water 사용량 할당
        target_test_result = pd.DataFrame(columns=self.region_B,index=self.target_total[start:end].index)
        for j in range(n_clusters):
            for k in list(set(self.region_B) & set(self.target_total.columns[labels==j])):
                target_test_result[k] = self.target_total[list(set(self.region_A) & set(self.target_cluster.columns[labels==j]))][start:end].mean(axis=1)

        #클러스터 구성요소가 B단지 1가구인 클러스터는 A단지 평균으로 할당
        for k in set(target_test_result.columns) - set(target_test_result.dropna(axis=1).columns):
            target_test_result[k] = self.target_total[self.region_A][start:end].mean(axis=1)

        return target_test_result

    def clustering_self_supervised(self, method, clustering, start, end):
        """
        method = gae or rae or tae
        """
        idx = int((start - self.train_index)/14)
        n_clusters = 5
        path = '/root/workspace/AMI/InferProj'
        if method == 'gae':
            repr_file_name =  f'h_r_{idx}_0'
        else:
            repr_file_name =  f'arr_{idx}_0'
#         arr_hidden = pd.read_csv(f'{path}/hidden/{method}_{self.infer_target}/{repr_file_name}.csv',index_col=0)
        arr_hidden = pd.read_csv(f'{path}/hidden/{method}_{self.infer_target}_2/{repr_file_name}.csv',index_col=0)

        pca = PCA(n_components=28*2, random_state=42)
        arr_hidden_pca = pca.fit_transform(arr_hidden)
    
        if clustering == 'kmedoids':
            
            clusters = KMedoids(n_clusters=n_clusters, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'kmeans':
            clusters = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit(arr_hidden_pca)
        elif clustering == 'agglomerative':
            clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(arr_hidden_pca)
        elif clustering == 'gmm':
            gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(arr_hidden_pca)
            clusters = type('GMMClusters', (object,), {'labels_': gmm.predict(arr_hidden_pca)})()

        #각 클러스터의 중심으로 water 사용량 할당
        labels = clusters.labels_
        target_test_result = pd.DataFrame(columns=self.region_B,index=self.target_total[start:end].index)
        for j in range(n_clusters):
            for k in list(set(self.region_B) & set(self.target_total.columns[labels==j])):
                target_test_result[k] = self.target_total[list(set(self.region_A) & set(self.target_cluster.columns[labels==j]))][start:end].mean(axis=1)

        #클러스터 구성요소가 B단지 1가구인 클러스터는 A단지 평균으로 할당
        for k in set(target_test_result.columns) - set(target_test_result.dropna(axis=1).columns):
            target_test_result[k] = self.target_total[self.region_A][start:end].mean(axis=1)

        return target_test_result

    def predict(self, model='rf', dist='euclidean', rep_method='gae', cl_method='kmedoids'):
        df_total_result = pd.DataFrame(columns=['4종 유추', '4종 랜덤', '4종 실제', '3종'])
        repr = []
        for week in range(1,6):
            # 1주일 간격
            t = 14*(week-1)

            # 에너지별 실제값
            cluster_A_sum = self.energy_A[self.region_B][self.train_index+t:self.test_index+t].sum(axis=1)
            cluster_B_sum = self.energy_B[self.region_B][self.train_index+t:self.test_index+t].sum(axis=1)
            cluster_C_sum = self.energy_C[self.region_B][self.train_index+t:self.test_index+t].sum(axis=1)
            
            # 실제 타겟값
            cluster_target_sum = self.target_total[self.region_B][self.train_index+t:self.test_index+t].sum(axis=1)

            # 타겟값 유추
            # if mode == 'kmedoids':
            #     fdm = self.FDM(self.train_index+t,self.test_index+t, dist)
            #     target_test_result = self.clustering(fdm, self.train_index+t,self.test_index+t)
            # else:
            target_test_result = self.clustering_self_supervised(rep_method, cl_method, self.train_index+t, self.test_index+t)
            pred_target_sum = target_test_result.sum(axis=1)
            pred_target_sum.index = cluster_target_sum.index
            repr.append(pred_target_sum)
            
            # 랜덤 유추값 생성
            random_target_sum = []
            for _ in range(56):
                random_target_sum.append(uniform(5,40))
            random_target_sum = pd.DataFrame(random_target_sum, index=pred_target_sum.index)

            # 데이터프레임 생성
            df = pd.concat([cluster_A_sum, cluster_B_sum, cluster_C_sum, cluster_target_sum, pred_target_sum, random_target_sum],axis=1)
            df.columns = self.exp_list[3] + [self.infer_target] + ['inferred','random']

            result_forecast_MAE = []
            result_forecast_MAPE = []
            for exp_list in self.exp_list:
                X = df[exp_list]
                y = df[[self.pred_target]].shift(-1)

                # 학습 및 테스트 데이터 분리
                train_size = int(len(X) * 0.7)
                train_X, test_X = X[:train_size], X[train_size:-1]
                train_y, test_y = y[:train_size], y[train_size:-1]

                scaler = MinMaxScaler()
                train_X = pd.DataFrame(scaler.fit_transform(train_X))
                test_X = pd.DataFrame(scaler.transform(test_X))

                if model == 'rf':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model == 'svr':
                    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                elif model == 'xgb':
                    model = xgb.XGBRegressor(seed=42)

                model.fit(train_X, train_y.values.ravel())
                    
                # 예측 수행
                predictions = model.predict(test_X)
                result_forecast_MAE.append(mean_absolute_error(test_y,predictions).round(3))
                result_forecast_MAPE.append(mean_absolute_percentage_error(test_y,predictions).round(4))
                
            df_result = pd.DataFrame(index=['MAE', 'MAPE'], columns=['4종 유추', '4종 랜덤', '4종 실제', '3종'])
            df_result.loc['MAE'] = result_forecast_MAE
            df_result.loc['MAPE'] = result_forecast_MAPE

            df_total_result.loc[f'Week {week}~{week+3}'] = df_result.loc['MAE']
        
        print(df_total_result)
        return df_total_result, repr
