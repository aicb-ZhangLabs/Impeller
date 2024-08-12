#import seurat in R
library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(SeuratWrappers)
library(anndata)
library(Matrix)
library(caTools)
library(dplyr)

benchmark_samples <- c(
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
)

data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC"
data_modes <- c("all_gene","high_variable_gene", )
for (data_mode in data_modes) {
    for (sample_number in benchmark_samples){
        test_mses <- c()
        val_mses <- c()
        print("----STARTING NEW RUN----")
        print(paste0("sample_number:", sample_number, " data_mode:", data_mode))
        for (split_version in 0:9){
            h5ad_path = paste0(data_dir, "/", sample_number, "/filtered_", sample_number, ".h5ad")
            adata <- anndata::read_h5ad(h5ad_path)

            library(reticulate)

            # Ensure numpy is available via reticulate
            np <- import("numpy")

            # Create the paths
            split_dir <- paste0(data_dir, "/DataSplit/", sample_number, "/", data_mode)

            # Test mask path
            test_mask_path <- paste0(split_dir, "/split_", split_version, "_test_mask.npz")

            # Validation mask path
            val_mask_path <- paste0(split_dir, "/split_", split_version, "_val_mask.npz")

            # Load the masks
            test_mask <- np$load(test_mask_path)$arr_0
            val_mask <- np$load(val_mask_path)$arr_0

            adata_train <- adata
            adata_test <- adata
            adata_val <- adata

            test_indices <- which(test_mask != 0, arr.ind = TRUE)
            val_indices <- which(val_mask != 0, arr.ind = TRUE)

            if (data_mode == "high_variable_gene"){
                test_indices <- test_indices[which(test_indices[,1] <= 3000 & test_indices[,2] <= 3000),]
                val_indices <- val_indices[which(val_indices[,1] <= 3000 & val_indices[,2] <= 3000),]
            }

            adata_train$X[test_indices] <- 0
            adata_train$X[val_indices] <- 0

            if (data_mode == "high_variable_gene"){
                adata <- adata[,adata$var[['highly_variable']]]
            }

            # create seurat object
            seurat_obj <- CreateSeuratObject(counts = t(adata_train$X), meta.data = adata_train$obs)

            # Load the Seurat library
            library(Seurat)

            # Find variable features
            seurat_obj <- FindVariableFeatures(seurat_obj)

            # Scale the data
            seurat_obj <- ScaleData(seurat_obj)

            # Perform linear dimensional reduction
            seurat_obj <- RunPCA(seurat_obj)

            # Run non-linear dimensional reduction (UMAP/tSNE)
            seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)

            # Cluster the cells
            seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
            seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

            # Now we can perform the imputation
            # Seurat uses a method called 'ALRA' for imputation
            seurat_obj <- RunALRA(seurat_obj)

            #list seurat_obj assays
            Assays(seurat_obj)

            #extract imputed data
            imputed_data <- GetAssayData(object = seurat_obj, assay = "alra", slot = "data")

            #compute MSE between test and original data
            print("--------")
            print(paste0("split_version:", split_version))
            print("ALARA test MSE")
            mse <- mean((imputed_data[test_indices] - adata$X[test_indices])^2)
            print(paste0("mse:", mse))
            print("ALARA val MSE")
            mse <- mean((imputed_data[val_indices] - adata$X[val_indices])^2)
            print(paste0("mse:", mse))
            test_mses <- c(test_mses, mse)
            val_mses <- c(val_mses, mse)
        

        }
        print(paste("sample_number:", sample_number, " data_mode:", data_mode))
        print(paste("Test MSE average:", mean(test_mses)))
        print(paste("Test MSE std:", sd(test_mses)))
        print(paste("Test MSE range:", range(test_mses)))
        print(paste("Val MSE average:", mean(val_mses)))
        print(paste("Val MSE std:", sd(val_mses)))
        print(paste("Val MSE range:", range(val_mses)))
        print("----FINISHED RUN----")

    }
}
