import os
import numpy as np
import scanpy as sc
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

import torch

from torch_geometric.data import Data


def prepare_data(data_platform: str = "10XVisium", sample_number: int = 151507, split_number: int = 0, \
                    preprocessed: bool = True, impute_gene_type: str = "all", \
                    use_spatial_graph: bool = False, spatial_graph_type: str = "radius", \
                    spatial_graph_radius_cutoff: int = 150, spatial_graph_knn_cutoff: int = 10, \
                    use_gene_graph: bool = False, gene_graph_knn_cutoff: int = 10, gene_graph_num_high_var_genes: int = 100, \
                    use_heterogeneous_graph: bool = False, \
                    ):

    if data_platform == "10XVisium":
        # At least one type of graph must be created
        assert (use_spatial_graph or use_gene_graph) == True
        if use_heterogeneous_graph:
            assert (use_spatial_graph and use_gene_graph) == True
        # Specify the data directory
        data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC"
        if preprocessed:
            adata = sc.read_h5ad(data_dir+"/Preprocessed/"+str(sample_number)+"/filtered_adata.h5ad")
            # Because we need to use id_cell_trans and cell_to_index in the spatial/gene graph
            spatial_graph_coor = pd.DataFrame(adata.obsm['spatial'])
            spatial_graph_coor.index = adata.obs.index
            spatial_graph_coor.columns = ['imagerow', 'imagecol']
            id_cell_trans = dict(zip(range(spatial_graph_coor.shape[0]), np.array(spatial_graph_coor.index)))
            cell_to_index = {id_cell_trans[idx]: idx for idx in id_cell_trans}
            # For spatial graph
            if use_spatial_graph:
                if spatial_graph_type == "radius":
                    nbrs = NearestNeighbors(radius=spatial_graph_radius_cutoff).fit(spatial_graph_coor)
                    distances, indices = nbrs.radius_neighbors(spatial_graph_coor, return_distance=True)
                elif spatial_graph_type == "knn":
                    nbrs = NearestNeighbors(n_neighbors=spatial_graph_knn_cutoff).fit(spatial_graph_coor)
                    distances, indices = nbrs.kneighbors(spatial_graph_coor, return_distance=True)
                else:
                    raise NotImplementedError
                spatial_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
                spatial_graph_KNN_df = pd.concat(spatial_graph_KNN_list)
                spatial_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
                Spatial_Net = spatial_graph_KNN_df.copy()
                Spatial_Net = Spatial_Net[Spatial_Net['Distance']>0]
                Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
                Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
                print(f'The spatial graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
                print(f'{Spatial_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
                adata.uns['Spatial_Net'] = Spatial_Net
                spatial_graph_edge_index = []
                for idx, row in Spatial_Net.iterrows():
                    cell1 = cell_to_index[row['Cell1']]
                    cell2 = cell_to_index[row['Cell2']]
                    spatial_graph_edge_index.append([cell1, cell2])
                    spatial_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
                spatial_graph_edge_index = torch.tensor(spatial_graph_edge_index, dtype=torch.long).t().contiguous()
            # For gene graph
            if use_gene_graph:
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=gene_graph_num_high_var_genes)
                hvg_indices = adata.var['highly_variable']
                hvg_data = torch.from_numpy(adata.X.todense().astype(np.float32))[:, hvg_indices]
                nbrs_gene = NearestNeighbors(n_neighbors=gene_graph_knn_cutoff).fit(hvg_data)
                distances, indices = nbrs_gene.kneighbors(hvg_data, return_distance=True)
                gene_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
                gene_graph_KNN_df = pd.concat(gene_graph_KNN_list)
                gene_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
                Gene_Net = gene_graph_KNN_df.copy()
                Gene_Net['Cell1'] = Gene_Net['Cell1'].map(id_cell_trans)
                Gene_Net['Cell2'] = Gene_Net['Cell2'].map(id_cell_trans)
                print(f'The gene graph contains {Gene_Net.shape[0]} edges, {adata.n_obs} cells.')
                print(f'{Gene_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
                gene_graph_edge_index = []
                for idx, row in Gene_Net.iterrows():
                    cell1 = cell_to_index[row['Cell1']]
                    cell2 = cell_to_index[row['Cell2']]
                    gene_graph_edge_index.append([cell1, cell2])
                    gene_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
                gene_graph_edge_index = torch.tensor(gene_graph_edge_index, dtype=torch.long).t().contiguous()
              
            # Create a PyTorch tensor for node features
            x = torch.from_numpy(adata.X.todense().astype(np.float32))
            original_x = x.clone()  # Save the original features
            val_mask = np.load(data_dir+"/Preprocessed/"+str(sample_number)+"/split_"+str(split_number)+"_val_mask.npz")['arr_0']
            test_mask = np.load(data_dir+"/Preprocessed/"+str(sample_number)+"/split_"+str(split_number)+"_test_mask.npz")['arr_0']
            val_mask = torch.tensor(val_mask)
            test_mask = torch.tensor(test_mask)
            x[val_mask] = 0
            x[test_mask] = 0
            
            if use_spatial_graph and not use_gene_graph:
                data = Data(x=x, edge_index=spatial_graph_edge_index)
            elif not use_spatial_graph and use_gene_graph:
                data = Data(x=x, edge_index=gene_graph_edge_index)
            elif use_spatial_graph and use_gene_graph:
                if use_heterogeneous_graph:
                    edge_index = torch.cat([spatial_graph_edge_index, gene_graph_edge_index], dim=1)
                    edge_type = torch.cat([torch.zeros(spatial_graph_edge_index.size(1), dtype=torch.long),
                                            torch.ones(gene_graph_edge_index.size(1), dtype=torch.long)], dim=0)
                    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
                else:
                    edge_index = torch.cat([spatial_graph_edge_index, gene_graph_edge_index], dim=1)
                    data = Data(x=x, edge_index=edge_index)
                    
            return data, val_mask, test_mask, x, original_x
        
        else:
            pass
            # sample_dir = f"{data_dir}/{sample_number}"
            # spatial_dir = f"{sample_dir}/spatial"

            # # If the spatial directory does not exist, create it
            # if not os.path.exists(spatial_dir):
            #     os.makedirs(spatial_dir)
            #     # Copy the tissue_hires_image.png file to the spatial directory
            #     os.system(f"cp {sample_dir}/tissue_hires_image.png {spatial_dir}")
            #     # and tissue_lowres_image.png
            #     os.system(f"cp {sample_dir}/tissue_lowres_image.png {spatial_dir}")
            #     # and scalefactors_json.json
            #     os.system(f"cp {sample_dir}/scalefactors_json.json {spatial_dir}")
            #     # Copy tissue_positions_list.txt but rename it to spatial/tissue_positions.csv
            #     os.system(f"cp {sample_dir}/tissue_positions_list.txt {spatial_dir}/tissue_positions_list.csv")
            #     # Give the spatial directory +777 permissions
            #     os.system(f"chmod 777 {spatial_dir}")

            # # Read the Visium data
            # adata = sc.read_visium(path=sample_dir, count_file=f"{sample_number}_filtered_feature_bc_matrix.h5")
            # adata.var_names_make_unique()

            # # Normalize the data
            # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            # sc.pp.normalize_total(adata, target_sum=1e4)
            # sc.pp.log1p(adata)

            # # Read the annotation
            # # Ann_df = pd.read_csv(f"{sample_dir}_truth.txt", sep='\t', header=None, index_col=0)
            # # Ann_df.columns = ['Ground Truth']

            # coor = pd.DataFrame(adata.obsm['spatial'])
            # coor.index = adata.obs.index
            # coor.columns = ['imagerow', 'imagecol']
            
            # if graph == "radius":
            #     nbrs = NearestNeighbors(radius=radius_cutoff).fit(coor)
            #     distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
            # elif graph == "knn":
            #     nbrs = NearestNeighbors(n_neighbors=knn_cutoff).fit(coor)
            #     distances, indices = nbrs.kneighbors(coor, return_distance=True)
            # else:
            #     raise NotImplementedError

            # KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
            # KNN_df = pd.concat(KNN_list)
            # KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

            # Spatial_Net = KNN_df.copy()
            # Spatial_Net = Spatial_Net[Spatial_Net['Distance']>0]
            # id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
            # Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
            # Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

            # print(f'The spatial graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
            # print(f'{Spatial_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')

            # adata.uns['Spatial_Net'] = Spatial_Net

            # cell_to_index = {id_cell_trans[idx]: idx for idx in id_cell_trans}

            # edge_index_spatial = []
            # for idx, row in Spatial_Net.iterrows():
            #     cell1 = cell_to_index[row['Cell1']]
            #     cell2 = cell_to_index[row['Cell2']]
            #     edge_index_spatial.append([cell1, cell2])
            #     # edge_index_spatial.append([cell2, cell1])  # Add reversed edge since the graph is undirected

            # edge_index_spatial = torch.tensor(edge_index_spatial, dtype=torch.long).t().contiguous()
            
            # # Create a PyTorch tensor for node features
            # x = torch.from_numpy(adata.X.todense().astype(np.float32))
            # original_x = x.clone()  # Save the original features

            # val_mask = np.load(data_dir+"/DataSplit/"+str(sample_number)+"/"+impute_gene_type+"_gene/split_"+str(split_number)+"_val_mask.npz")['arr_0']
            # test_mask = np.load(data_dir+"/DataSplit/"+str(sample_number)+"/"+impute_gene_type+"_gene/split_"+str(split_number)+"_test_mask.npz")['arr_0']
            
            # val_mask = torch.tensor(val_mask)
            # test_mask = torch.tensor(test_mask)
            
            # if impute_gene_type == "high_variable":
            #     x = x[:, adata.var.highly_variable]
            #     original_x = original_x[:, adata.var.highly_variable]
            #     val_mask = val_mask[:, adata.var.highly_variable]
            #     test_mask = test_mask[:, adata.var.highly_variable]
            #     x[val_mask] = 0
            #     x[test_mask] = 0
            # else:
            #     x[val_mask] = 0
            #     x[test_mask] = 0

            # if het:
            #     hvg_indices = adata.var['highly_variable']

            #     hvg_data = torch.from_numpy(adata.X.todense().astype(np.float32))[:, hvg_indices]
            #     nbrs_gene = NearestNeighbors(n_neighbors=gene_cutoff).fit(hvg_data)
            #     distances_gene, indices_gene = nbrs_gene.kneighbors(hvg_data, return_distance=True)

            #     KNN_list_gene = [pd.DataFrame(zip([it]*indices_gene[it].shape[0], indices_gene[it], distances_gene[it])) for it in range(indices_gene.shape[0])]
            #     KNN_df_gene = pd.concat(KNN_list_gene)
            #     KNN_df_gene.columns = ['Cell1', 'Cell2', 'Distance']

            #     Gene_Net = KNN_df_gene.copy()
            #     Gene_Net['Cell1'] = Gene_Net['Cell1'].map(id_cell_trans)
            #     Gene_Net['Cell2'] = Gene_Net['Cell2'].map(id_cell_trans)
                
            #     print(f'The gene graph contains {Gene_Net.shape[0]} edges, {adata.n_obs} cells.')
            #     print(f'{Gene_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')

            #     edge_index_gene = []
            #     for idx, row in Gene_Net.iterrows():
            #         cell1 = cell_to_index[row['Cell1']]
            #         cell2 = cell_to_index[row['Cell2']]
            #         edge_index_gene.append([cell1, cell2])
            #         # edge_index_gene.append([cell2, cell1])  # Add reversed edge since the graph is undirected

            #     edge_index_gene = torch.tensor(edge_index_gene, dtype=torch.long).t().contiguous()

            #     edge_index = torch.cat([edge_index_spatial, edge_index_gene], dim=1)

            #     edge_type = torch.cat([torch.zeros(edge_index_spatial.size(1), dtype=torch.long),
            #                         torch.ones(edge_index_gene.size(1), dtype=torch.long)], dim=0)

            #     data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
            # else:
            #     data = Data(x=x, edge_index=edge_index_spatial)

            # return data, val_mask, test_mask, x, original_x
    else:
        raise NotImplementedError
