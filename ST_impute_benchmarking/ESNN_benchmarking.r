#import seurat in R
library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(SeuratWrappers)
library(anndata)
library(Matrix)
library(caTools)
library(dplyr)
library(reticulate)


benchmark_samples = c(
    'one sample'
)

Ks = c(
    10,30,60,90,120
)

#seed for reproducibility
set.seed(1)

data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/Stereoseq/MouseOlfactoryBulb/Preprocessed"
data_modes <- c("all_gene")
for (data_mode in data_modes) {
    for (sample_number in benchmark_samples){
        for (k in Ks){
            sample_dir = data_dir
            test_mses <- c()
            val_mses <- c()
            test_l1_distances <- c()
            val_l1_distances <- c()
            test_cosine_sims <- c()
            val_cosine_sims <- c()
            test_rmses <- c()
            val_rmses <- c()
            print("----STARTING NEW RUN----")
            print(paste0("sample_number:", sample_number, " data_mode:", data_mode))
            for (split_version in 0:3){
                h5ad_path = paste0(data_dir, "/filtered_adata.h5ad")
                adata <- anndata::read_h5ad(h5ad_path)


                np <- import("numpy")
                # Create the paths
                split_dir <- sample_dir

                # Test mask path
                test_mask_path <- paste0(split_dir, "/split_", split_version, "_test_mask.npz")

                # Validation mask path
                val_mask_path <- paste0(split_dir, "/split_", split_version, "_val_mask.npz")

                # Load the masks
                test_mask <- np$load(test_mask_path)$arr_0
                val_mask <- np$load(val_mask_path)$arr_0


                py_copy <- import("copy")
                adata_train <- py_copy$deepcopy(adata)
                
                test_indices <- which(test_mask != 0, arr.ind = TRUE)
                val_indices <- which(val_mask != 0, arr.ind = TRUE)

                adata_train$X[test_indices] <- 0
                adata_train$X[val_indices] <- 0

                # create seurat object
                seurat_obj <- CreateSeuratObject(counts = adata_train$X, meta.data = adata_train$obs)

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
                seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10, k.param=k)
                seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

                # Now you can use the spatial KNN graph for imputation
                impute_spatial_snn <- function(seurat_obj) {
                    # Use Seurat's accessor functions to get the spatial SNN graph and data
                    spatial_snn <- seurat_obj@graphs$RNA_snn
                    data <- seurat_obj@assays$RNA@counts
                    
                    # Convert data to dense matrix if it's sparse
                    if (inherits(data, "sparseMatrix")) {
                        data <- as.matrix(data)
                    }
                    
                    # Identify genes with zero expression across all cells
                    zero_genes_mat <- data == 0
                    
                    # For each cell, get the indices of its shared nearest neighbors
                    snn_indices <- apply(spatial_snn != 0, 2, which)
                    
                    # Ensure snn_indices is a matrix
                    if (is.list(snn_indices)) {
                        max_len <- max(sapply(snn_indices, length))
                        snn_matrix <- matrix(NA, length(snn_indices), max_len)
                        for (i in 1:length(snn_indices)) {
                        snn_matrix[i, 1:length(snn_indices[[i]])] <- snn_indices[[i]]
                        }
                        snn_indices <- snn_matrix
                    }
                    
                    # Vectorized computation of weighted mean of SNN for zero genes
                    imputed_values <- matrix(0, nrow = nrow(data), ncol = ncol(data))
                    for (i in 1:ncol(data)) {
                        zero_genes <- which(zero_genes_mat[, i])
                        current_snn <- snn_indices[i, !is.na(snn_indices[i, ])]
                        neighbors_data <- data[zero_genes, current_snn, drop = FALSE]
                        neighbors_weights <- spatial_snn[i, current_snn]
                        imputed_means <- apply(neighbors_data, 1, function(x) weighted.mean(x, w = neighbors_weights, na.rm = TRUE))
                        
                        # Replace NA values with 0
                        imputed_means[is.na(imputed_means)] <- 0
                        
                        imputed_values[zero_genes, i] <- imputed_means
                    }
                    
                    # Replace zero values in data with imputed values
                    data[zero_genes_mat] <- imputed_values[zero_genes_mat]
                    
                    # Update the original data with the imputed data using Seurat's accessor functions
                    seurat_obj <- Seurat::SetAssayData(seurat_obj, assay = "RNA", slot = "counts", new.data = data)
                    
                    return(seurat_obj)
                }


                # Use the function to impute the missing data
                seurat_obj <- impute_spatial_snn(seurat_obj)
                imputed_data <- seurat_obj@assays$RNA@counts
                imputed_data[is.nan(imputed_data)] <- 0
                # print(dim(imputed_data))
                # print(dim(adata$X))
                # print(dim(test_mask))
                # print(dim(val_mask))
                #compute MSE between test and original data
                print("--------")
                print(paste0("split_version:", split_version))
                print("Spatial Expression KNN test MSE")
                test_mse <- mean((imputed_data[test_indices] - adata$X[test_indices])^2)
                print(paste0("mse:", test_mse))
                print("Spatial Expresion KNN val MSE")
                val_mse <- mean((imputed_data[val_indices] - adata$X[val_indices])^2)
                print(paste0("mse:", val_mse))
                test_l1_distance <- mean(abs(imputed_data[test_indices] - adata$X[test_indices]))
                val_l1_distance <- mean(abs(imputed_data[val_indices] - adata$X[val_indices]))
                # For test data
                norm_imputed_test <- sqrt(sum(imputed_data[test_indices]^2))
                norm_adata_test <- sqrt(sum(adata$X[test_indices]^2))
                dot_product_test <- sum(imputed_data[test_indices] * adata$X[test_indices])
                test_cosine_sim <- 1 - (dot_product_test / (norm_imputed_test * norm_adata_test))

                # For validation data
                norm_imputed_val <- sqrt(sum(imputed_data[val_indices]^2))
                norm_adata_val <- sqrt(sum(adata$X[val_indices]^2))
                dot_product_val <- sum(imputed_data[val_indices] * adata$X[val_indices])
                val_cosine_sim <- 1 - (dot_product_val / (norm_imputed_val * norm_adata_val))
                test_rmse <- sqrt(mean((imputed_data[test_indices] - adata$X[test_indices])^2))
                val_rmse <- sqrt(mean((imputed_data[val_indices] - adata$X[val_indices])^2))
                print("Spatial Expression KNN test L1 distance")
                print(paste0("L1 distance:", test_l1_distance))
                print("Spatial Expression KNN val L1 distance")
                print(paste0("L1 distance:", val_l1_distance))
                print("Spatial Expression KNN test cosine similarity")
                print(paste0("cosine similarity:", test_cosine_sim))
                print("Spatial Expression KNN val cosine similarity")
                print(paste0("cosine similarity:", val_cosine_sim))
                print("Spatial Expression KNN test RMSE")
                print(paste0("RMSE:", test_rmse))
                print("Spatial Expression KNN val RMSE")
                print(paste0("RMSE:", val_rmse))
                test_l1_distances <- c(test_l1_distances, test_l1_distance)
                val_l1_distances <- c(val_l1_distances, val_l1_distance)
                test_cosine_sims <- c(test_cosine_sims, test_cosine_sim)
                val_cosine_sims <- c(val_cosine_sims, val_cosine_sim)
                test_rmses <- c(test_rmses, test_rmse)
                val_rmses <- c(val_rmses, val_rmse)
                test_mses <- c(test_mses, test_mse)
                val_mses <- c(val_mses, val_mse)
                
                
            

            }
        print(paste("----FINISHED RUN----"))
        print(paste("K value:", k))
        print(paste("sample_number:", sample_number, " data_mode:", data_mode))
        print(paste("Test MSE average:", mean(test_mses)))
        print(paste("Test MSE std:", sd(test_mses)))
        print(paste("Val MSE average:", mean(val_mses)))
        print(paste("Val MSE std:", sd(val_mses)))
        print(paste("Test L1 distance average:", mean(test_l1_distances)))
        print(paste("Test L1 distance std:", sd(test_l1_distances)))
        print(paste("Val L1 distance average:", mean(val_l1_distances)))
        print(paste("Val L1 distance std:", sd(val_l1_distances)))
        print(paste("Test cosine similarity average:", mean(test_cosine_sims)))
        print(paste("Test cosine similarity std:", sd(test_cosine_sims)))
        print(paste("Val cosine similarity average:", mean(val_cosine_sims)))
        print(paste("Val cosine similarity std:", sd(val_cosine_sims)))
        print(paste("Test RMSE average:", mean(test_rmses)))
        print(paste("Test RMSE std:", sd(test_rmses)))
        print(paste("Val RMSE average:", mean(val_rmses)))
        print(paste("Val RMSE std:", sd(val_rmses)))
        print("----FINISHED RUN----")
        }
    }
}
