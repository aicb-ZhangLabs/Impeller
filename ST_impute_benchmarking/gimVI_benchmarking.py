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
# import squidpy as sq
import PIL
import numpy as np
import anndata as ad
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import scipy

benchmark_samples = [
    151507,
    151508,
    151509,
    151669,
    151670,
    151671,
    151672,
    151673,
    151674,
    151675,
    151676
]

for lantent_value in [30, 60]:
    for sample_number in benchmark_samples:
        for data_mode in ["high_variable_gene"]:
            test_mses = []
            val_mses = []
            print("----STARTING NEW RUN----")
            print("Params: " , "Sample", sample_number, "lantent_value: ", lantent_value)
            for split_version in range(0,10):
                data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC"
                sample_dir = data_dir + "/" + str(sample_number)
                spatial_dir = sample_dir + "/spatial"

                #if spatial directory does not exist, create it
                if not os.path.exists(spatial_dir):
                    os.makedirs(spatial_dir)
                    #copy the tissue_hires_image.png file to the spatial directory
                    os.system("cp " + sample_dir + "/tissue_hires_image.png " + spatial_dir)
                    #and tissue_lowres_image.png
                    os.system("cp " + sample_dir + "/tissue_lowres_image.png " + spatial_dir)
                    #and scalefactors_json.json
                    os.system("cp " + sample_dir + "/scalefactors_json.json " + spatial_dir)
                    #copy tissue_positions_list.txt but rename it to spatial/tissue_positions.csv
                    os.system("cp " + sample_dir + "/tissue_positions_list.txt " + spatial_dir + "/tissue_positions_list.csv")
                    #give spatial direction +777 permissions
                    os.system("chmod -R 777 " + spatial_dir)

                adata = sc.read_visium(sample_dir,count_file=f"{sample_number}_filtered_feature_bc_matrix.h5")
                adata.var_names_make_unique()
                adata.raw = adata.copy()
                adata.var["mt"] = adata.var_names.str.startswith("MT-")
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

                # # Filter Data

                #
                # Normalize the data
                sc.pp.normalize_total(adata)
                # Log transform the data
                sc.pp.log1p(adata)
                # Identify highly variable genes
                sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

                # [markdown]


                #
                split_dir = data_dir + "/DataSplit/" +  str(sample_number) + "/" + data_mode
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

                # # Highly variable selection

                #
                if data_mode == "high_variable_gene":
                    adata = adata.copy()[:, adata.var["highly_variable"]] # the copy here is for gimVI to work

                #  [markdown]
                # # Build Train Adata

                # [markdown]
                # # gimVI
                # has spatial context

                # 
                from scvi.external import GIMVI
                adata = adata_train.copy()
                spatial_adata = adata_train.copy()


                #  [markdown]
                # If the below code give you some error about adata not matching you just have to clear notebook memory and start again. It's do to how they are managing memory in their implementation

                # 
                GIMVI.setup_anndata(adata)
                GIMVI.setup_anndata(spatial_adata)

                gimVi_spatial = spatial_adata
                gimVi_expression = adata

                model = GIMVI(gimVi_expression, gimVi_spatial, n_latent=lantent_value)

                # 
                model.train()
                # 
                _, imputed_values = model.get_imputed_values()
                imputed_values = scipy.sparse.csr_matrix(imputed_values)
                # 

                imputed_sparse = csr_matrix(imputed_values)
                adata_sparse = csr_matrix(adata.X)
                test_mse = np.square(imputed_sparse[test_mask] - adata_sparse[test_mask]).mean()
                val_mse = np.square(imputed_sparse[val_mask] - adata_sparse[val_mask]).mean()

                print("----------------")
                print("Data: ", "sample_number: ", sample_number, "data_mode: ", data_mode, "split_version: ", split_version)
                print("Params: " , "lantent_value: ", lantent_value)
                print("------")
                print("gimVI MSE for test mask", test_mse)
                print("------")
                print("------")
                print("gimVI MSE for val mask", val_mse)
                print("------")
                test_mses.append(test_mse)
                val_mses.append(val_mse)
            print("Params: " , "Sample", sample_number, "lantent_value: ", lantent_value)
            print("Test MSE average: ", np.mean(test_mses))
            print("Test MSE std: ", np.std(test_mses))
            print("Test MSE range: ", np.max(test_mses) - np.min(test_mses))
            print("Val MSE average: ", np.mean(val_mses))
            print("Val MSE std: ", np.std(val_mses))
            print("Val MSE range: ", np.max(val_mses) - np.min(val_mses))
            print("----ENDING RUN----")
            





