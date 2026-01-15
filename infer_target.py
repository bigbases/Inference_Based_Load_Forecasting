# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import trange

import sklearn
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids, CommonNNClustering, CLARA

from dtaidistance import ed
from scipy.cluster.hierarchy import fcluster, linkage

from dataproc import DataProc

# 경로 설정
path = '/root/workspace/AMI/InferProj'

# 미수집 에너지 유추: 클러스터링 및 유추
class InferTarget:
    def __init__(self, target, data_type, n_clusters=5):
        self.n_clusters = n_clusters
        self.target = target
        self.data_type = data_type

        dp = DataProc(target)
        dp.preprocess()
        self.dp = dp

        # 사용할 데이터 범위 설정
        if data_type == "test":
            self.start_idx = dp.train_index
            self.end_idx = dp.test_index
        elif data_type == "cluster":
            self.start_idx = dp.test_index
            self.end_idx = dp.cluster_index

        # 변수 저장
        self.region_A = dp.region_A
        self.region_B = dp.region_B
        self.target_total = dp.target_total
        self.target_cluster = dp.target_cluster
        self.energy_total = dp.energy_total
        self.energy_cluster = dp.energy_cluster
        self.elec_total = dp.elec_total
        
    # A단지 평균: 클러스터링없이 평균해 예측
    def regionA_average(self):
        error_MAE = sklearn.metrics.mean_absolute_error(self.target_total[self.region_B][self.start_idx:self.end_idx], 
                                                        np.tile(self.target_total[self.region_A][self.start_idx:self.end_idx].mean(axis=1)[:, np.newaxis], (1, len(self.region_B))))
        return error_MAE
    
    # 클러스터링: 전기 사용량
    def cluster_by_elec(self, clustering):
        elec = self.elec_total[self.start_idx:self.end_idx]
        sc = MinMaxScaler()
        elec = pd.DataFrame(sc.fit_transform(elec))
        data_for_cluster = np.transpose(elec)

        if clustering == 'kmedoids':
            clusters = KMedoids(n_clusters=self.n_clusters, random_state=42).fit(data_for_cluster)
        elif clustering == 'cnnc':
            clusters = CommonNNClustering(eps=1, min_samples=3, algorithm='auto', leaf_size=30).fit(data_for_cluster)
        elif clustering == 'clara':
            clusters = CLARA(n_clusters=self.n_clusters, n_sampling_iter=5, random_state=42).fit(data_for_cluster)

        si = sklearn.metrics.silhouette_score(data_for_cluster.squeeze(),labels=clusters.labels_)

        return clusters, si
    
    # 클러스터링: Fused Distance Matrix
    def cluster_by_fdm(self, clustering):
        norm_list = []
        for i in self.energy_total:
            fdm = i[self.start_idx:self.end_idx]
            rows = []
            for j in range(fdm.shape[1]):
                row = []
                for k in range(fdm.shape[1]):
                    row.append(ed.distance(fdm[:,j],fdm[:,k]))
                rows.append(row)
            dist = np.array(rows)
            dist = pd.DataFrame(dist)
            norm = dist/dist.mean().mean()
            norm_list.append(norm)

        fdm = norm_list[0] + norm_list[1] + norm_list[2]
    
        if clustering == 'kmedoids':
            clusters = KMedoids(n_clusters=self.n_clusters, random_state=42, metric='precomputed').fit(fdm)
        elif clustering == 'cnnc':
            clusters = CommonNNClustering(eps=0.5, min_samples=5, metric='precomputed', algorithm='auto', leaf_size=30).fit(fdm)
        elif clustering == 'clara':
            clusters = CLARA(n_clusters=self.n_clusters, metric='precomputed', n_sampling_iter=5, random_state=42).fit(fdm)

        si = sklearn.metrics.silhouette_score(fdm, labels=clusters.labels_, metric='precomputed')

        return fdm, clusters, si
    
    # 클러스터링: 단순 concat
    def cluster_by_md(self, mode):
        # 단순 concatenate
        n_clusters=5
        for n, i in enumerate(self.energy_total):
            energy = i[self.start_idx:self.end_idx]
            energy = MinMaxScaler().fit_transform(energy)
            if n == 0:
                data = energy
            else:
                data = np.r_[data, energy]
        data = data.T
    
        if mode == 'kmeans':
            clusters = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        elif mode == 'kmedoids':
            clusters = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean').fit(data)

        si = sklearn.metrics.silhouette_score(data, labels=clusters.labels_)

        return data, clusters, si
    
    # 클러스터링: LSTM AutoEncoder
    def cluster_by_RAE(self, clustering, num, dim=28*2):
        
        arr_hidden = pd.read_csv(f'{path}/hidden/rae_{self.target}/arr_hidden_{self.data_type}_12H_{num}.csv',index_col=0)

        pca = PCA(n_components=dim, random_state=42)
        arr_hidden_pca = pca.fit_transform(arr_hidden)
    
        if clustering == 'kmedoids':
            clusters = KMedoids(n_clusters=self.n_clusters, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'cnnc':
            clusters = CommonNNClustering(eps=0.5, min_samples=5, algorithm='auto', leaf_size=30).fit(arr_hidden_pca)
        elif clustering == 'clara':
            clusters = CLARA(n_clusters=self.n_clusters, n_sampling_iter=5, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'kmeans':
            clusters = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42).fit(arr_hidden_pca)
        elif clustering == 'dbscan':
            clusters = DBSCAN(eps=0.5, min_samples=5).fit(arr_hidden_pca)
        elif clustering == 'agglomerative':
            clusters = AgglomerativeClustering(n_clusters=self.n_clusters).fit(arr_hidden_pca)
        elif clustering == 'spectral':
            clusters = SpectralClustering(n_clusters=self.n_clusters, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'gmm':
            gmm = GaussianMixture(n_components=self.n_clusters, random_state=42).fit(arr_hidden_pca)
            clusters = type('GMMClusters', (object,), {'labels_': gmm.predict(arr_hidden_pca)})()

        si = sklearn.metrics.silhouette_score(arr_hidden_pca,clusters.labels_)

        return clusters, si
    
    # 클러스터링: Graph AutoEncoder
    def cluster_by_GAE(self, clustering, num, dim=28*2):
        
        # arr_hidden = pd.read_csv(f'{path}/hidden/gae_{self.target}/h_r_{self.data_type}_12H_10{num}.csv',index_col=0)
        arr_hidden = pd.read_csv(f'{path}/hidden/gae_{self.target}/h_r_{num}_0.csv',index_col=0)

        pca = PCA(n_components=dim, random_state=42)
        arr_hidden_pca = pca.fit_transform(arr_hidden)
    
        if clustering == 'kmedoids':
            clusters = KMedoids(n_clusters=self.n_clusters, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'cnnc':
            clusters = CommonNNClustering(eps=0.5, min_samples=5, algorithm='auto', leaf_size=30).fit(arr_hidden_pca)
        elif clustering == 'clara':
            clusters = CLARA(n_clusters=self.n_clusters, n_sampling_iter=5, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'kmeans':
            clusters = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42).fit(arr_hidden_pca)
        elif clustering == 'dbscan':
            clusters = DBSCAN(eps=0.5, min_samples=5).fit(arr_hidden_pca)
        elif clustering == 'agglomerative':
            clusters = AgglomerativeClustering(n_clusters=self.n_clusters).fit(arr_hidden_pca)
        elif clustering == 'spectral':
            clusters = SpectralClustering(n_clusters=self.n_clusters, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'gmm':
            gmm = GaussianMixture(n_components=self.n_clusters, random_state=42).fit(arr_hidden_pca)
            clusters = type('GMMClusters', (object,), {'labels_': gmm.predict(arr_hidden_pca)})()

        si = sklearn.metrics.silhouette_score(arr_hidden_pca,clusters.labels_)

        return clusters, si
    
    # 클러스터링: Transformer AutoEncoder
    def cluster_by_TAE(self, clustering, num, dim=28*2):
        
        arr_hidden = pd.read_csv(f'{path}/hidden/tae_{self.target}/arr_hidden_{self.data_type}_12H_{num}.csv',index_col=0)

        pca = PCA(n_components=dim, random_state=42)
        arr_hidden_pca = pca.fit_transform(arr_hidden)
    
        if clustering == 'kmedoids':
            clusters = KMedoids(n_clusters=self.n_clusters, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'cnnc':
            clusters = CommonNNClustering(eps=0.5, min_samples=5, algorithm='auto', leaf_size=30).fit(arr_hidden_pca)
        elif clustering == 'clara':
            clusters = CLARA(n_clusters=self.n_clusters, n_sampling_iter=5, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'kmeans':
            clusters = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42).fit(arr_hidden_pca)
        elif clustering == 'dbscan':
            clusters = DBSCAN(eps=0.5, min_samples=5).fit(arr_hidden_pca)
        elif clustering == 'agglomerative':
            clusters = AgglomerativeClustering(n_clusters=self.n_clusters).fit(arr_hidden_pca)
        elif clustering == 'spectral':
            clusters = SpectralClustering(n_clusters=self.n_clusters, random_state=42).fit(arr_hidden_pca)
        elif clustering == 'gmm':
            gmm = GaussianMixture(n_components=self.n_clusters, random_state=42).fit(arr_hidden_pca)
            clusters = type('GMMClusters', (object,), {'labels_': gmm.predict(arr_hidden_pca)})()

        si = sklearn.metrics.silhouette_score(arr_hidden_pca,clusters.labels_)

        return clusters, si
    
    # 타겟 예측: 훈련된 클러스터링 모델 객체를 사용해 클러스터링한 뒤 예측 진행
    def predict_target_original(self, kmedoids, si):
        labels = kmedoids.labels_

        #각 클러스터의 중심으로 사용량 할당
        target_test_result = pd.DataFrame(columns=self.region_B,index=self.target_total[self.start_idx:self.end_idx].index)
        for j in range(self.n_clusters):
            for k in list(set(self.region_B) & set(self.target_total.columns[labels==j])):
                target_test_result[k] = self.target_total[list(set(self.region_A) & set(self.target_cluster.columns[labels==j]))][self.start_idx:self.end_idx].mean(axis=1)

        #클러스터 구성요소가 B단지 1가구인 클러스터는 A단지 평균으로 할당
        for k in set(target_test_result.columns) - set(target_test_result.dropna(axis=1).columns):
            target_test_result[k] = self.target_total[self.region_A][self.start_idx:self.end_idx].mean(axis=1)

        error_MAE = sklearn.metrics.mean_absolute_error(self.target_total[self.region_B][self.start_idx:self.end_idx].sum(axis=1), target_test_result[self.region_B].sum(axis=1))
        # print('MAE :',error_MAE.round(3))
        # print('SI :',si.round(3))
        return error_MAE
    
    # 타겟 예측: 훈련된 클러스터링 모델 객체를 사용해 클러스터링한 뒤 예측 진행
    def predict_target(self, kmedoids, si):
        labels = kmedoids.labels_

        #각 클러스터의 중심으로 사용량 할당
        target_test_result = pd.DataFrame(columns=self.region_B,index=self.target_total[self.start_idx:self.end_idx].index)
        for j in range(self.n_clusters):
            for k in list(set(self.region_B) & set(self.target_total.columns[labels==j])):
                target_test_result[k] = self.target_total[list(set(self.region_A) & set(self.target_cluster.columns[labels==j]))][self.start_idx:self.end_idx].mean(axis=1)

        #클러스터 구성요소가 B단지 1가구인 클러스터는 A단지 평균으로 할당
        for k in set(target_test_result.columns) - set(target_test_result.dropna(axis=1).columns):
            target_test_result[k] = self.target_total[self.region_A][self.start_idx:self.end_idx].mean(axis=1)
            
        true = self.target_total[self.region_B][self.start_idx:self.end_idx]
        infer = target_test_result[self.region_B]

        error_MAE = sklearn.metrics.mean_absolute_error(true, infer)
        # print('MAE :',error_MAE.round(3))
        # print('SI :',si.round(3))
        return true, infer, error_MAE

    def predict_target_knn(self, n_neighbors=5):
        """
        KNN 기반 cross-building 추론 baseline.
        - 임베딩 사용 X
        - 원본 시계열(target_total)만 사용
        - 출력: true, infer, error_MAE (기존 predict_target과 동일 형식)
        """
        df = self.target_total  # index: time, columns: region_A + region_B
        time_index = df.index

        # 테스트 기간
        test_idx = slice(self.start_idx, self.end_idx)
        test_index = df.iloc[self.start_idx:self.end_idx].index

        # --- 1) history 구간 선택 (유사도 계산에 사용할 구간) ---
        hist = df.iloc[:self.start_idx]

        # region_A / region_B 분리
        A_cols = list(self.region_A)
        B_cols = list(self.region_B)

        # history 구간에서 A/B 시계열 추출
        hist_A = hist[A_cols]  # (T_hist, |A|)
        hist_B = hist[B_cols]  # (T_hist, |B|)

        # --- 2) B 가구별로, A 가구들과의 유사도(상관 기반 거리) 계산 ---
        # corr(B, A): hist_A.T.corr(hist_B.T) 를 써도 되지만,
        # 여기선 직접 구현 대신 pandas corr를 활용 (각 B에 대해 한 번씩).

        # 추론 결과를 담을 DF (test 구간 × B 가구)
        target_test_result = pd.DataFrame(columns=B_cols, index=test_index, dtype=float)

        for b in B_cols:
            # B 가구의 history
            y = hist_B[b]

            # 상수 시계열이면 유사도 계산이 의미 없으니, A 평균으로 fallback
            if y.nunique() <= 1:
                # 나중에 전체 A 평균으로 처리 (kNN 의미 X)
                continue

            # 각 A 가구와의 피어슨 상관계수 계산
            # (행: 시간, 열: A 가구들)
            corr = hist_A.corrwith(y)

            # NaN (상수 시계열 등) 제거
            corr = corr.dropna()
            if len(corr) == 0:
                # 상관계수 계산 안 되면, 나중에 fallback
                continue

            # 상관계수 → 거리: d = 1 - corr
            dist = 1.0 - corr
            # 거리가 작은 순으로 k개 선택
            knn_cols = dist.nsmallest(min(n_neighbors, len(dist))).index.tolist()

            # --- 3) test 구간에서 이웃 A 가구들의 시간별 평균으로 B 추론 ---
            target_test_result[b] = df.loc[test_index, knn_cols].mean(axis=1)

        # --- 4) 유사도 계산 실패(빈 컬럼)는 A 전체 평균으로 채우기 (AVG fallback) ---
        # NaN만 있는 컬럼들 찾기
        nan_cols = [c for c in target_test_result.columns
                    if target_test_result[c].isna().all()]

        if len(nan_cols) > 0:
            # A 전체 평균 (test 구간 기준)
            avg_A = df.loc[test_index, A_cols].mean(axis=1)
            for c in nan_cols:
                target_test_result[c] = avg_A

        # --- 5) MAE 계산 (ground truth vs inference) ---
        true = df.loc[test_index, B_cols]
        infer = target_test_result[B_cols]

        error_MAE = mean_absolute_error(true.values.flatten(), infer.values.flatten())

        return true, infer, error_MAE


    def predict_target_matrix_completion(self, n_neighbors=5):
        df = self.target_total.copy()
        time_index = df.index
        test_index = df.iloc[self.start_idx:self.end_idx].index

        A_cols = list(self.region_A)
        B_cols = list(self.region_B)

        # Ground truth
        true = df.loc[test_index, B_cols].copy()

        # test 구간의 B만 NaN으로 가리기
        df_masked = df.copy()
        df_masked.loc[test_index, B_cols] = np.nan

        # (time, building) 그대로 sample×feature 로 사용
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        arr_imputed = imputer.fit_transform(df_masked.values)
        df_imputed = pd.DataFrame(arr_imputed, index=time_index, columns=df.columns)

        infer = df_imputed.loc[test_index, B_cols]
        error_MAE = mean_absolute_error(true.values.flatten(), infer.values.flatten())

        return true, infer, error_MAE
