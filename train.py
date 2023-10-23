import warnings
warnings.filterwarnings("ignore")

import copy
import argparse

import torch

from data import prepare_data
from model import PathGNN, MultiLayerGCN, MultiLayerGAT, MultiLayerGraphSAGE, MultiLayerTransformer, MultiLayerGIN, MultiLayerRGCN, STAGATE
from utils import NeighborSampler, evaluate_with_batch, evaluate_pathgnn, get_paths

def train(args):
    
    data, val_mask, test_mask, x, original_x = prepare_data(data_platform=args.data_platform, sample_number=args.sample_number, split_number=args.split_number, \
                                                            preprocessed=args.preprocessed, impute_gene_type=args.impute_gene_type, \
                                                            use_spatial_graph=args.use_spatial_graph, \
                                                            spatial_graph_type=args.spatial_graph_type, \
                                                            spatial_graph_radius_cutoff=args.spatial_graph_radius_cutoff, \
                                                            spatial_graph_knn_cutoff=args.spatial_graph_knn_cutoff, \
                                                            use_gene_graph=args.use_gene_graph, \
                                                            gene_graph_knn_cutoff=args.gene_graph_knn_cutoff, \
                                                            gene_graph_num_high_var_genes=args.gene_graph_num_high_var_genes, \
                                                            use_heterogeneous_graph=args.use_heterogeneous_graph
                                                            )
    
    if args.model == "GCN":
        model = MultiLayerGCN(feature_size=data.num_node_features, hidden_size=args.hidden_size, output_size=data.num_node_features, num_layers=args.num_layers)
    elif args.model == "GAT":
        model = MultiLayerGAT(feature_size=data.num_node_features, hidden_size=args.hidden_size, output_size=data.num_node_features, num_layers=args.num_layers)
    elif args.model == "GIN":
        model = MultiLayerGIN(feature_size=data.num_node_features, hidden_size=args.hidden_size, output_size=data.num_node_features, num_layers=args.num_layers)
    elif args.model == "GraphSAGE":
        model = MultiLayerGraphSAGE(feature_size=data.num_node_features, hidden_size=args.hidden_size, output_size=data.num_node_features, num_layers=args.num_layers)
    elif args.model == "GraphTransformer":
        model = MultiLayerTransformer(feature_size=data.num_node_features, hidden_size=args.hidden_size, output_size=data.num_node_features, num_layers=args.num_layers)
    elif args.model == "STAGATE":
        model = STAGATE(feature_size=data.num_node_features, hidden_size=args.hidden_size, output_size=args.hidden_size)
    elif args.model == "RGCN":
        assert args.use_heterogeneous_graph == True
        model = MultiLayerRGCN(feature_size=data.num_node_features, hidden_size=args.hidden_size, output_size=data.num_node_features, num_relations=2, num_layers=args.num_layers)
    elif args.model == "PathGNN":
        model = PathGNN(in_dim=data.num_node_features, hidden_dim=args.hidden_size, out_dim=data.num_node_features, dropout=args.dropout,
                    num_layers=args.num_layers, num_paths=args.num_paths, path_length=args.path_length, num_edge_types=args.num_edge_types,\
                    alpha=args.alpha, operator_type=args.operator_type)
    else:
        raise NotImplementedError

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    x = x.to(device)
    original_x = original_x.to(device)

    print(f'Model structure: {model}, \n\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if args.model == "PathGNN":
        optimizer = model.setup_optimizer(args.lr, args.weight_decay, args.lr_oc, args.wd_oc)
        paths, path_types = get_paths(args, data)
        data = data.to(device)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    max_no_improvement_epochs = args.patience
    best_model_state = None  # To store the state of the best model
    
    if args.model != "PathGNN":
        if args.use_heterogeneous_graph:
            sampler = NeighborSampler(edge_index=data.edge_index, sizes=[-1]*args.num_layers, edge_type=data.edge_type, \
                                        device=device, batch_size=args.batch_size, shuffle=True)
            sampler_eval = NeighborSampler(edge_index=data.edge_index, sizes=[-1]*args.num_layers, edge_type=data.edge_type, \
                                        device=device, batch_size=x.shape[0], shuffle=False)
        else:
            sampler = NeighborSampler(edge_index=data.edge_index, sizes=[-1]*args.num_layers, edge_type=None, \
                                        device=device, batch_size=args.batch_size, shuffle=True)
            sampler_eval = NeighborSampler(edge_index=data.edge_index, sizes=[-1]*args.num_layers, edge_type=None, \
                                        device=device, batch_size=1, shuffle=False)
    
    for epoch in range(args.epochs):
        model.train()
        
        if args.model == "PathGNN":
            optimizer.zero_grad()
            logits = model(x, paths, path_types)
            if args.fit_type == "all":
                loss = criterion(logits, x)
            else:
                target_x = x
                # Create a boolean mask of the target where True indicates non-zero values
                mask = target_x != 0
                # Apply the mask to select only the non-zero targets and predictions
                target_x_masked = target_x[mask]
                target_logits_masked = logits[mask]
                # Compute the loss only over the non-zero values
                loss = criterion(target_logits_masked, target_x_masked)
            loss.backward()
            optimizer.step()
            val_l1_distance, val_cosine_sim, val_rmse = evaluate_pathgnn(args, model, x, paths, path_types, criterion, device, val_mask, original_x)
        else:
            for batch_size, n_id, adjs in sampler:
                optimizer.zero_grad()
                x = data.x[n_id].to(device)  # Node feature matrix
                if args.num_layers == 1:
                    adjs = [adjs.to(device)]
                    edge_indices = [adj.edge_index for adj in adjs]
                else:
                    adjs = [adj.to(device) for adj in adjs]
                    edge_indices = [adj.edge_index for adj in adjs]
                # Compute the loss and gradients, and update the model parameters
                if args.model == "RGCN":
                    edge_types = [adj.edge_type for adj in adjs]
                    logits = model(x, edge_indices, edge_types)
                else:
                    logits = model(x, edge_indices)
                target_logits = logits[:batch_size]
                target_x = x[:batch_size]
                if args.fit_type == "all":
                    loss = criterion(target_logits, target_x)
                else:
                    # Create a boolean mask of the target where True indicates non-zero values
                    mask = target_x != 0
                    # Apply the mask to select only the non-zero targets and predictions
                    target_x_masked = target_x[mask]
                    target_logits_masked = target_logits[mask]
                    # Compute the loss only over the non-zero values
                    loss = criterion(target_logits_masked, target_x_masked)
                loss.backward()
                optimizer.step()
            val_l1_distance, val_cosine_sim, val_rmse = evaluate_with_batch(args, model, sampler_eval, data, criterion, device, val_mask, original_x)
        if epoch % args.print_epoch == 0: 
            print(f"Epoch: {epoch}, Loss: {loss.item()}, val_l1_distance: {val_l1_distance}, val_cosine_sim: {val_cosine_sim}, val_rmse: {val_rmse}")
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            no_improvement_epochs = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement_epochs += 1
        if no_improvement_epochs >= max_no_improvement_epochs:
            print(f"No improvement in validation loss for {max_no_improvement_epochs} epochs, stopping.")
            break
    # Load the best model state before testing
    model.load_state_dict(best_model_state)
    if args.model == "PathGNN":
        test_l1_distance, test_cosine_sim, test_rmse = evaluate_pathgnn(args, model, x, paths, path_types, criterion, device, test_mask, original_x)
    else:
        test_l1_distance, test_cosine_sim, test_rmse = evaluate_with_batch(args, model, sampler, data, criterion, device, test_mask, original_x)
#     print(f"Final MSE: {test_loss}")

    print(f"Final l1_distance: {test_l1_distance}, test_cosine_sim: {test_cosine_sim}, test_rmse: {test_rmse}")
    
    return test_l1_distance, test_cosine_sim, test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # For the data platform and sample
    parser.add_argument('--data_platform', type=str, default='10XVisium')
    parser.add_argument('--sample_number', type=int, default=151507)
    parser.add_argument('--split_number', type=int, choices=list(range(10)), default=0)
    
    # For the preprocessed method
    parser.add_argument('--preprocessed', type=bool, default=True)
    parser.add_argument('--impute_gene_type', type=str, choices=['all', 'high_variable'], default='high_variable')
    
    # For the graph construction
    # spatial graph
    parser.add_argument('--use_spatial_graph', type=bool, default=True)
    # parser.add_argument('--use_spatial_graph', type=bool, default=False)
    parser.add_argument('--spatial_graph_type', type=str, choices=['radius', 'knn'], default='radius')
    parser.add_argument('--spatial_graph_radius_cutoff', type=int, default=150)
    parser.add_argument('--spatial_graph_knn_cutoff', type=int, default=10)
    # gene graph
    parser.add_argument('--use_gene_graph', type=bool, default=True)
    # parser.add_argument('--use_gene_graph', type=bool, default=False)
    parser.add_argument('--gene_graph_knn_cutoff', type=int, default=5)
    parser.add_argument('--gene_graph_num_high_var_genes', type=int, default=100)
    # heterogeneous graph
    parser.add_argument('--use_heterogeneous_graph', type=bool, default=True)
    # parser.add_argument('--use_heterogeneous_graph', type=bool, default=False)
    
    # Model architecture
    parser.add_argument('--model', type=str, default='PathGNN')
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_paths', type=int, default=10)
    parser.add_argument('--path_length', type=int, default=8)
    parser.add_argument('--num_edge_types', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--operator_type', type=str, choices=['global', 'shared_layer', 'shared_channel', 'independent'], default='independent')
    
    parser.add_argument('--spatial_walk_p', type=int, default=1)
    parser.add_argument('--spatial_walk_q', type=int, default=1)
    parser.add_argument('--gene_walk_p', type=int, default=1)
    parser.add_argument('--gene_walk_q', type=int, default=1)
    
    # parser.add_argument('--model', type=str, default='GCN')
    # parser.add_argument('--model', type=str, default='GIN')
    # parser.add_argument('--model', type=str, default='STAGATE')
    # parser.add_argument('--model', type=str, default='RGCN')
    parser.add_argument('--hidden_size', type=int, default=64)
    
    # Training process
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--lr_oc", type=float, default=1e-2)
    parser.add_argument("--wd_oc", type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--print_epoch', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--fit_type', type=str, choices=['all', 'nonzero'], default='nonzero')
     
    args = parser.parse_args()
    train(args)
