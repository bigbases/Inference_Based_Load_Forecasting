import pandas as pd
import numpy as np
from time import time

# 커스텀 모듈 import
from infer_target import InferTarget

cluster_methods = ['kmeans']
n_clusters=5
cluster_results = []
test_results = []
for n_clusters in [5]:
    for infer_target in ['gas', 'hotwater', 'water']:
        target_cluster = InferTarget(infer_target, 'cluster', n_clusters=n_clusters)
        target_test = InferTarget(infer_target, 'test', n_clusters=n_clusters)

        # AVG
        avg_cluster_mean = target_cluster.regionA_average()
        avg_test_mean = target_test.regionA_average()

        print('\n[AVG]')
        print(f'Cluster MAE mean: {avg_cluster_mean}')
        print(f'Test MAE mean: {avg_test_mean}')

        # KNN
        knn_cluster_mean = target_cluster.predict_target_knn()[-1]
        knn_test_mean = target_test.predict_target_knn()[-1]

        print('\n[KNN]')
        print(f'Cluster MAE mean: {knn_cluster_mean}')
        print(f'Test MAE mean: {knn_test_mean}')

        # Matrix Completion
        mc_cluster_mean = target_cluster.predict_target_matrix_completion()[-1]
        mc_test_mean = target_test.predict_target_matrix_completion()[-1]

        print('\n[Matrix Completion]')
        print(f'Cluster MAE mean: {mc_cluster_mean}')
        print(f'Test MAE mean: {mc_test_mean}')

        # RAE
        rae_cluster = []
        rae_test = []
        for cluster_method in cluster_methods:
            mae_cluster = []
            mae_test = []
            for i in range(101, 104):
                cluster, si = target_cluster.cluster_by_RAE(cluster_method,i)
                mae = target_cluster.predict_target(cluster, si)[-1]
                mae_cluster.append(mae)
                cluster, si = target_test.cluster_by_RAE(cluster_method,i)
                mae = target_test.predict_target(cluster, si)[-1]
                mae_test.append(mae)
            
            print(f'\n[RAE - {cluster_method}]')
            print('cluster MAE mean:',(c_mean:=np.array(mae_cluster).mean()).round(10))
            print('cluster MAE std:',(c_std:=np.array(mae_cluster).std()).round(10))
            print('test MAE mean:',(t_mean:=np.array(mae_test).mean()).round(10))
            print('test MAE std:',(t_std:=np.array(mae_test).std()).round(10))
            print()
            
            rae_cluster.append(c_mean)
            rae_test.append(t_mean)

        # GAE
        gae_cluster = []
        gae_test = []
        for cluster_method in cluster_methods:
            mae_cluster = []
            mae_test = []
            for i in range(1, 5):
                cluster, si = target_cluster.cluster_by_GAE(cluster_method,i)
                mae = target_cluster.predict_target(cluster, si)[-1]
                mae_cluster.append(mae)
                cluster, si = target_test.cluster_by_GAE(cluster_method,i)
                mae = target_test.predict_target(cluster, si)[-1]
                mae_test.append(mae)
            
            print(f'\n[GAE - {cluster_method}]')
            print('cluster MAE mean:',(c_mean:=np.array(mae_cluster).mean()).round(10))
            print('cluster MAE std:',(c_std:=np.array(mae_cluster).std()).round(10))
            print('test MAE mean:',(t_mean:=np.array(mae_test).mean()).round(10))
            print('test MAE std:',(t_std:=np.array(mae_test).std()).round(10))
            print()
            
            gae_cluster.append(c_mean)
            gae_test.append(t_mean)

        # TAE
        tae_cluster = []
        tae_test = []
        for cluster_method in cluster_methods:
            mae_cluster = []
            mae_test = []
            for i in range(101, 105):
                cluster, si = target_cluster.cluster_by_TAE(cluster_method,i)
                mae = target_cluster.predict_target(cluster, si)[-1]
                mae_cluster.append(mae)
                cluster, si = target_test.cluster_by_TAE(cluster_method,i)
                mae = target_test.predict_target(cluster, si)[-1]
                mae_test.append(mae)
            
            print(f'\n[TAE - {cluster_method}]')
            print('cluster MAE mean:',(c_mean:=np.array(mae_cluster).mean()).round(10))
            print('cluster MAE std:',(c_std:=np.array(mae_cluster).std()).round(10))
            print('test MAE mean:',(t_mean:=np.array(mae_test).mean()).round(10))
            print('test MAE std:',(t_std:=np.array(mae_test).std()).round(10))
            print()
            
            tae_cluster.append(c_mean)
            tae_test.append(t_mean)
        
        cluster_results.append([n_clusters, infer_target, avg_cluster_mean, knn_cluster_mean, mc_cluster_mean]+rae_cluster+gae_cluster+tae_cluster)
        test_results.append([n_clusters, infer_target, avg_test_mean, knn_test_mean, mc_test_mean]+rae_test+gae_test+tae_test)

df_cluster = pd.DataFrame(cluster_results, columns=['N_Clusters', 'Infer Target', 'AVG', 'KNN', 'MC', 
                                                    'RAE KMedoids', 'RAE KMeans', 'RAE Agglomerative', 'RAE GMM',
                                                    'GAE KMedoids', 'GAE KMeans', 'GAE Agglomerative', 'GAE GMM',
                                                    'TAE KMedoids', 'TAE KMeans', 'TAE Agglomerative', 'TAE GMM'])

df_test = pd.DataFrame(test_results, columns=['N_Clusters', 'Infer Target', 'AVG', 'KNN', 'MC', 
                                                    'RAE KMedoids', 'RAE KMeans', 'RAE Agglomerative', 'RAE GMM',
                                                    'GAE KMedoids', 'GAE KMeans', 'GAE Agglomerative', 'GAE GMM',
                                                    'TAE KMedoids', 'TAE KMeans', 'TAE Agglomerative', 'TAE GMM'])

df_cluster.to_csv('./exp_result/cluster_results.csv', index=False)
df_test.to_csv('./exp_result/test_results.csv', index=False)