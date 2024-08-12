#  [markdown]
# # Load Libraries

# 
import torch
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from matplotlib import rcParams
import PIL
import numpy as np
import anndata as ad
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import stlearn as st

benchmark_samples = [
    151507,
    151508,
    151509,
    151669,
    151510,
    151669,
    151670,
    151671,
    151672,
    151673,
    151674,
    151675,
    151676
]

for lantent_value in [30]:
    for sample_number in benchmark_samples:
        for data_mode in ["high_variable_gene"]:
            test_mses = []
            val_mses = []
            test_l1s = []
            val_l1s = []
            test_cos_sims = []
            val_cos_sims = []
            test_rmses = []
            val_rmses = []
            print("----STARTING NEW RUN----")
            print("Params: " , "Sample", sample_number, "lantent_value: ", lantent_value)
            for split_version in range(0,4):
                data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC/Preprocessed"
                sample_dir = data_dir + "/" + str(sample_number)
                base_dir = f"/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC/{sample_number}"
                spatial_dir = base_dir + "/spatial"

                file_path = sample_dir + "/filtered_adata_before_norm.h5ad"
                adata = sc.read_h5ad(file_path)
                adata.var_names_make_unique()
                adata.raw = adata.copy()
                adata.var["mt"] = adata.var_names.str.startswith("MT-")
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

                # # Filter Data
                # Data is already filtered

                #
                split_dir = sample_dir
                #test mask path
                test_mask_path = split_dir + "/split_" + str(split_version) + "_test_mask.npz"
                #val mask path
                val_mask_path = split_dir + "/split_" + str(split_version) + "_val_mask.npz"
                #load masks
                test_mask = np.load(test_mask_path)["arr_0"]
                val_mask = np.load(val_mask_path)["arr_0"]


                #
                adata_train = adata.copy()
                # mask adata train with test mask and val mask, true's become 0
                adata_train.X[test_mask] = 0
                adata_train.X[val_mask] = 0

                adata_test = adata.copy()
                adata_test.X[test_mask] = 0

                adata_val = adata.copy()
                adata_val.X[val_mask] = 0

                #  [markdown]
                # # Build Train Adata

                # [markdown]
                # # gimVI
                # has spatial context

                # 
                st_adata_train = st.convert_scanpy(adata_train)
                img_train = spatial_dir + "/tissue_hires_image.png"
                st.add.image(st_adata_train, img_train, library_id='image')

                st.pp.neighbors(st_adata_train, n_neighbors=30)
                st.pp.tiling(st_adata_train)
                st.pp.extract_feature(st_adata_train)
            
                st.em.run_pca(st_adata_train)

                st.spatial.SME.SME_impute0(st_adata_train)


                import copy
                #get true indices from test mask in a column
                adata_impute = copy.deepcopy(adata)
                adata_impute.X = copy.deepcopy(st_adata_train.obsm['imputed_data'])

                #normalize both adata
                sc.pp.normalize_total(adata_impute, target_sum=1e4)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata_impute)
                sc.pp.log1p(adata)
                #

                imputed_sparse = csr_matrix(adata_impute.X)
                adata_sparse = csr_matrix(adata.X)

                test_mse = np.square(imputed_sparse[test_mask] - adata_sparse[test_mask]).mean()
                val_mse = np.square(imputed_sparse[val_mask] - adata_sparse[val_mask]).mean()
                test_l1 = np.abs(imputed_sparse[test_mask] - adata_sparse[test_mask]).mean()
                val_l1 = np.abs(imputed_sparse[val_mask] - adata_sparse[val_mask]).mean()
                test_cos_sim = cosine_similarity(np.asarray(imputed_sparse[test_mask]), np.asarray(adata_sparse[test_mask]))
                val_cos_sim = cosine_similarity(np.asarray(imputed_sparse[val_mask]), np.asarray(adata_sparse[val_mask]))
                test_rmse = np.sqrt(test_mse)
                val_rmse = np.sqrt(val_mse)

                print("----------------")
                print("Data: ", "sample_number: ", sample_number, "data_mode: ", data_mode, "split_version: ", split_version)
                print("Params: " , "lantent_value: ", lantent_value)
                print("------")
                print("gimVI MSE for test mask", test_mse)
                print("gimVI MSE for val mask", val_mse)
                print("gimVI L1 for test mask", test_l1)
                print("gimVI L1 for val mask", val_l1)
                print("gimVI Cosine Similarity for test mask", test_cos_sim)
                print("gimVI Cosine Similarity for val mask", val_cos_sim)
                print("gimVI RMSE for test mask", test_rmse)
                print("gimVI RMSE for val mask", val_rmse)
                print("------")
                test_mses.append(test_mse)
                val_mses.append(val_mse)
                test_l1s.append(test_l1)
                val_l1s.append(val_l1)
                test_cos_sims.append(test_cos_sim)
                val_cos_sims.append(val_cos_sim)
                test_rmses.append(test_rmse)
                val_rmses.append(val_rmse)

            print("Params: " , "Sample", sample_number, "lantent_value: ", lantent_value)
            print("Test MSE average: ", np.mean(test_mses))
            print("Test MSE std: ", np.std(test_mses))
            print("Test L1 average: ", np.mean(test_l1s))
            print("Test L1 std: ", np.std(test_l1s))
            print("Test Cosine Similarity average: ", np.mean(test_cos_sims))
            print("Test Cosine Similarity std: ", np.std(test_cos_sims))
            print("Test RMSE average: ", np.mean(test_rmses))
            print("Test RMSE std: ", np.std(test_rmses))
            print("Val MSE average: ", np.mean(val_mses))
            print("Val MSE std: ", np.std(val_mses))
            print("Val L1 average: ", np.mean(val_l1s))
            print("Val L1 std: ", np.std(val_l1s))
            print("Val Cosine Similarity average: ", np.mean(val_cos_sims))
            print("Val Cosine Similarity std: ", np.std(val_cos_sims))
            print("Val RMSE average: ", np.mean(val_rmses))
            print("Val RMSE std: ", np.std(val_rmses))
            print("----ENDING RUN----")
            




