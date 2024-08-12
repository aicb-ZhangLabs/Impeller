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
data_modes <- c("high_variable_gene", "all_gene")
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

            # Perform linear dimensional reduction PCA!!!!
            adata_train$obs <- cbind(adata_train$obs, adata_train$obsm[['spatial']])
            spatial_data <- cbind(adata_train$obs[["1"]], adata_train$obs[["2"]])
            spatial_matrix = matrix(spatial_data, nrow = nrow(adata_train), ncol = 2)
            expression_matrix = cbind(adata_train$X)
            #transpose the expression matrix
            #combine the two matrices
            combined_matrix <- cbind(expression_matrix, spatial_matrix)

            se_pca <- prcomp(combined_matrix)

            se_pca$x <- se_pca$x[,1:10]
            print("fraction")
            #insert the pca values into the seurat object
            seurat_obj@reductions$pca <- CreateDimReducObject(embeddings = se_pca$x, key = "combinedPC_")

            # Find variable features
            seurat_obj <- FindVariableFeatures(seurat_obj)

            # Scale the data
            seurat_obj <- ScaleData(seurat_obj)

            # # Run non-linear dimensional reduction (UMAP/tSNE)
            # seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)

            # # Cluster the cells this builds the knn
            seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10, k.param=20, reduction = "pca")
            # Now you can use the spatial KNN graph for imputation
            impute_spatial_knn <- function(seurat_obj) {
            # Get the spatial KNN graph
            spatial_knn <- seurat_obj@graphs$RNA_snn

            # Get the data
            data <- seurat_obj@assays$RNA@counts

            # Initialize a matrix to hold the imputed data
            imputed_data <- data # we start with the original data

            # Loop over each cell
            for (i in 1:ncol(data)) {
                # Get the gene expression values for a single cell
                cell_expression <- data[, i]

                # Check if any values are zero
                if (any(cell_expression == 0)) {
                # Get the K nearest neighbors for the cell
                knn <- spatial_knn[,i]
                # Find the genes that are not expressed in the cell (expression == 0)
                zero_genes <- cell_expression == 0
                # Get the expression of the K nearest neighbors for the genes that are zero
                neighbor_expression <- data[zero_genes, knn]
                # Check if neighbor_expression is a vector or a matrix
                if (is.matrix(neighbor_expression)) {
                    # Impute the missing values by averaging the expression values of the K nearest neighbors
                    imputed_values <- rowMeans(neighbor_expression, na.rm = TRUE)
                } else {
                    # If neighbor_expression is a vector, just use the values directly
                    imputed_values <- neighbor_expression
                }
                message(paste0("Imputed cell:", i))

                # Replace the zero expression with the imputed values
                cell_expression[zero_genes] <- imputed_values
                }

                # Store the imputed cell expression values
                imputed_data[, i] <- cell_expression
            }

            # Replace the original data with the imputed data
            seurat_obj@assays$RNA@counts <- imputed_data

            return(seurat_obj)
            }

            # Use the function to impute the missing data
            seurat_obj <- impute_spatial_knn(seurat_obj)
            imputed_data <- seurat_obj@assays$RNA@counts
            print("--------")
            print(paste0("split_version:", split_version))
            print("EKNN test MSE")
            mse <- mean((imputed_data[test_indices] - adata$X[test_indices])^2)
            print(paste0("mse:", mse))
            print("EKNN val MSE")
            mse <- mean((imputed_data[val_indices] - adata$X[val_indices])^2)
            print(paste0("mse:", mse))

        

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
