import dgl
import torch
import torch.nn.functional as F

def get_paths(args, data):
    device = args.device
    if args.use_heterogeneous_graph:
        # Separate the graph by edge type
        g0, g1 = separate_graph_by_edge_type(data.cpu())
        # Perform random walk on each graph
        path0 = get_random_walk_path(g0, args.num_paths, args.path_length-1, p=args.spatial_walk_p, q=args.spatial_walk_q)
        path1 = get_random_walk_path(g1, args.num_paths, args.path_length-1, p=args.gene_walk_p, q=args.gene_walk_q)
        paths = torch.vstack((path0,path1)).to(device).long()

        path0_types = torch.zeros(path0.shape[0], dtype=torch.long)
        path1_types = torch.ones(path1.shape[0], dtype=torch.long)
        path_types = torch.hstack((path0_types, path1_types)).to(device)
    else:
        if args.use_spatial_graph:
            paths = get_random_walk_path(pyg_to_dgl(data.cpu()), args.num_paths, args.path_length-1, p=args.spatial_walk_p, q=args.spatial_walk_q)
        else:
            paths = get_random_walk_path(pyg_to_dgl(data.cpu()), args.num_paths, args.path_length-1, p=args.gene_walk_p, q=args.gene_walk_q)
        paths = paths.to(device).long()
        path_types = torch.zeros(paths.shape[0], dtype=torch.long).to(device)
        
    return paths, path_types

def get_random_walk_path(g, num_walks, walk_length, p=1, q=1):
    """
    Get random walk paths.
    """
    device = g.device
    g = g.to("cpu")
    walks = []
    nodes = g.nodes()

    for _ in range(num_walks):
        walks.append(
            # dgl.sampling.random_walk(g, nodes, length=walk_length)[0]
            dgl.sampling.node2vec_random_walk(g, nodes, p=p, q=q, walk_length=walk_length)
        )
    walks = torch.stack(walks).to(device) # (num_walks, num_nodes, walk_length)
    return walks

def pyg_to_dgl(data):
    g = dgl.DGLGraph()
    g.add_nodes(data.num_nodes)
    g.add_edges(data.edge_index[0], data.edge_index[1])

    return g

def separate_graph_by_edge_type(data):
    mask_type_0 = data.edge_type == 0
    mask_type_1 = data.edge_type == 1
    
    edge_index_0 = data.edge_index[:, mask_type_0]
    edge_index_1 = data.edge_index[:, mask_type_1]
    
    g0 = dgl.DGLGraph()
    g0.add_nodes(data.num_nodes)
    g0.add_edges(edge_index_0[0], edge_index_0[1])
    
    g1 = dgl.DGLGraph()
    g1.add_nodes(data.num_nodes)
    g1.add_edges(edge_index_1[0], edge_index_1[1])

    return g0, g1

def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def evaluate_pathgnn(args, model, x, paths, path_types, criterion, device, mask, original_x):
    model.eval()
    total_loss = 0

    with torch.no_grad():

        # Compute the logits
        logits = model(x, paths, path_types)

        mask = mask.to(device)

        masked_logits = logits[mask]
        masked_original_x = original_x[mask]
        
        # loss = criterion(masked_logits, masked_original_x)
        # total_loss = loss.item() * mask.sum().item()

    # return total_loss / mask.sum().item()
    
    all_masked_logits = masked_logits
    all_masked_original_x = masked_original_x
    
    l1_distance = F.l1_loss(all_masked_logits, all_masked_original_x).item()
    cosine_sim = cosine_similarity(all_masked_logits, all_masked_original_x).mean().item()
    rmse = torch.sqrt(F.mse_loss(all_masked_logits, all_masked_original_x)).item()

    return l1_distance, cosine_sim, rmse

def evaluate_with_batch(args, model, sampler, data, criterion, device, mask, original_x):
    model.eval()
    total_loss = 0
    total_count = 0
    
    all_masked_logits = []
    all_masked_original_x = []

    with torch.no_grad():
        for batch_size, n_id, adjs in sampler:
            
            # Apply mask to get validation nodes
            mask_n_id = mask[n_id].to(device)
            mask_n_id = mask_n_id[:batch_size]

            if mask_n_id.sum().item() == 0:
                continue
            
            x = data.x[n_id].to(device)  # Node feature matrix

            masked_original_x = original_x[n_id].to(device)
            
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
            target_original_x = masked_original_x[:batch_size]

            masked_logits = target_logits[mask_n_id]
            masked_original_x = target_original_x[mask_n_id]
            
            # print("In evaluation:")
            # print("masked_logits", masked_logits)
            # print("masked_original_x", masked_original_x)

            # loss = criterion(masked_logits, masked_original_x)
            # total_loss += loss.item() * mask_n_id.sum().item()
            # total_count += mask_n_id.sum().item()
            
            all_masked_logits.append(masked_logits)
            all_masked_original_x.append(masked_original_x)

    # return total_loss / total_count
    all_masked_logits = torch.cat(all_masked_logits, dim=0)
    all_masked_original_x = torch.cat(all_masked_original_x, dim=0)

    l1_distance = F.l1_loss(all_masked_logits, all_masked_original_x).item()
    cosine_sim = cosine_similarity(all_masked_logits, all_masked_original_x).mean().item()
    rmse = torch.sqrt(F.mse_loss(all_masked_logits, all_masked_original_x)).item()

    return l1_distance, cosine_sim, rmse

from typing import List, Optional, Tuple, NamedTuple
from torch_sparse import SparseTensor


class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)

class Adj_het(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]
    edge_type: torch.Tensor

    def to(self, *args, **kwargs):
        return Adj_het(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size, self.edge_type.to(*args, **kwargs))


class NeighborSampler(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Given a GNN with :math:`L` layers and a specific mini-batch of nodes
    :obj:`node_idx` for which we want to compute embeddings, this module
    iteratively samples neighbors and constructs bipartite graphs that simulate
    the actual computation flow of GNNs.

    More specifically, :obj:`sizes` denotes how much neighbors we want to
    sample for each node in each layer.
    This module then takes in these :obj:`sizes` and iteratively samples
    :obj:`sizes[l]` for each node involved in layer :obj:`l`.
    In the next layer, sampling is repeated for the union of nodes that were
    already encountered.
    The actual computation graphs are then returned in reverse-mode, meaning
    that we pass messages from a larger set of nodes to a smaller one, until we
    reach the nodes for which we originally wanted to compute embeddings.

    Hence, an item returned by :class:`NeighborSampler` holds the current
    :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
    computation, and a list of bipartite graph objects via the tuple
    :obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
    bipartite edges between source and target nodes, :obj:`e_id` denotes the
    IDs of original edges in the full graph, and :obj:`size` holds the shape
    of the bipartite graph.
    For each bipartite graph, target nodes are also included at the beginning
    of the list of source nodes so that one can easily apply skip-connections
    or add self-loops.

    .. note::

        For an example of using :obj:`NeighborSampler`, see
        `examples/reddit.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        reddit.py>`_ or
        `examples/ogbn_products_sage.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_products_sage.py>`_.

    Args:
        edge_index (LongTensor): The edge indices of the full-graph.
        size ([int]): The number of neighbors to
            sample for each node in each layer. If set to :obj:`sizes[i] = -1`,
            all neighbors are included in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, edge_index: torch.Tensor, sizes: List[int],
                 edge_type: Optional[torch.Tensor] = None,
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target", device: str = "cpu", **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        edge_attr = torch.arange(edge_index.size(1)).to(device)
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (N, N),
                                           is_sorted=False)
        adj = adj.t() if flow == 'source_to_target' else adj
        self.adj = adj.to('cpu')
        
        self.edge_type = edge_type

        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero().view(-1)

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        super(NeighborSampler, self).__init__(node_idx.tolist(),
                                              collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            adj, n_id = self.adj.sample_adj(n_id, size, replace=False)
            row, col, e_id = adj.coo()
            size = adj.sparse_sizes()
            if self.flow == 'source_to_target':
                edge_index = torch.stack([col, row], dim=0)
                size = size[::-1]
            else:
                edge_index = torch.stack([row, col], dim=0)
            
            if self.edge_type!=None:
                edge_type = self.edge_type[e_id]
                adjs.append(Adj_het(edge_index, e_id, size, edge_type)) 
            else:
                adjs.append(Adj(edge_index, e_id, size)) 
            
        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)