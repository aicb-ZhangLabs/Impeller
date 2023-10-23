import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter

from typing import Union, Tuple, Optional

from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, TransformerConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch_geometric.nn import RGCNConv

import itertools

class PathGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers, num_paths, path_length, num_edge_types, alpha, operator_type="independent"):
        super(PathGNN, self).__init__()
        self._dropout = dropout
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc_in.weight, gain=1.414)
        self.in_act = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc_out.weight, gain=1.414)
        self.out_act = nn.ReLU()
        
        self.layers = nn.ModuleList([
                PathGNNLayer(hidden_dim, num_paths, path_length, num_edge_types)
                for _ in range(num_layers)
            ])
        
        if operator_type == 'global':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, 1)) for _ in range(num_edge_types)])])
        elif operator_type == 'shared_layer':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, hidden_dim)) for _ in range(num_edge_types)])])
        elif operator_type == 'shared_channel':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, 1)) for _ in range(num_edge_types)]) for _ in range(num_layers)])
        elif operator_type == 'independent':
            self.path_weights = nn.ModuleList([nn.ParameterList([nn.Parameter(torch.Tensor(1, path_length, hidden_dim)) for _ in range(num_edge_types)]) for _ in range(num_layers)])
                        
        for path_weight_layer in self.path_weights:
            for path_weight in path_weight_layer:
                nn.init.xavier_normal_(path_weight, gain=1.414)
        
        # self.layers = nn.ModuleList([
        #     PathGNNLayer(hidden_dim, num_paths, path_length, num_edge_types)
        #     for _ in range(num_layers)
        # ])
        
        self.num_layers = num_layers
        self.num_paths = num_paths
        self.path_length = path_length
        self.num_edge_types = num_edge_types
        self.alpha = alpha
        self.operator_type = operator_type
    
    def forward(self, input_x, paths, path_types):
        in_feats = F.dropout(input_x, p=self._dropout, training=self.training)
        in_feats = self.fc_in(in_feats)
        in_feats = self.in_act(in_feats)

        feats = in_feats
        for i in range(self.num_layers):
            if self.operator_type == "global" or self.operator_type == "shared_layer":
                feats = self.layers[i](feats, paths, path_types, self.path_weights[0])
            elif self.operator_type == "shared_channel" or self.operator_type == "independent":
                feats = self.layers[i](feats, paths, path_types, self.path_weights[i])
            else:
                raise NotImplementedError
            feats = self.alpha * in_feats + (1 - self.alpha) * feats

        feats = F.dropout(feats, p=self._dropout, training=self.training)
        out = self.fc_out(feats)
        out = self.out_act(out)
        return out

    def setup_optimizer(self, lr, wd, lr_oc, wd_oc):
        param_list = [
            {"params": self.layers.parameters(), "lr": lr, "weight_decay": wd},
            {"params": itertools.chain(*[self.fc_in.parameters(), self.fc_out.parameters()]), "lr": lr_oc, "weight_decay": wd_oc} 
        ]
        return torch.optim.Adam(param_list)
    
class PathGNNLayer(nn.Module):
    def __init__(self, hidden_dim, num_path, path_length, num_edge_types):
        super(PathGNNLayer, self).__init__()
        
        self.fc = nn.Linear(num_edge_types*hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.num_path = num_path
        self.path_length = path_length
        self.num_edge_types = num_edge_types

    def forward(self, feats, paths, path_types, path_weights):
        """
            feats: (num_nodes, d),
            paths: (num_path, num_nodes, path_length)
            path_types: (num_path,) contains the edge type of each path
        """
        results = []
        for edge_type, path_weight in enumerate(path_weights):
            mask = (path_types == edge_type) # select the paths of this type
            paths_of_type = paths[mask] # (num_paths_of_type, num_nodes, path_length)
            path_feats = feats[paths_of_type] # (num_paths_of_type, num_nodes, path_length, d)
            path_feats = (path_feats * path_weight).sum(dim=2) # (num_paths_of_type, num_nodes, d)
            path_feats = path_feats.mean(dim=0) # (num_nodes, d)
            results.append(path_feats)
        if self.num_edge_types == 2:
            fout = torch.hstack((results[0],results[1]))
        else:
            fout = results[0]

        fout = self.fc(fout)
        fout = F.relu(fout)
        return fout
    

class MultiLayerGCN(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(MultiLayerGCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers==1:
            self.layers.append(GCNConv(feature_size, output_size, flow="target_to_source"))
        else:
            self.layers.append(GCNConv(feature_size, hidden_size, flow="target_to_source"))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_size, hidden_size, flow="target_to_source"))
            self.layers.append(GCNConv(hidden_size, output_size, flow="target_to_source"))

    def forward(self, x, edge_indices):
        for i, edge_index in enumerate(edge_indices):
            new_edge_index = torch.vstack((edge_index[1,:],edge_index[0,:]))
            x = self.layers[i](x, new_edge_index)
            x = torch.relu(x)
            if i != len(edge_indices) - 1:  # Not apply dropout for the last layer
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class MultiLayerGAT(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(MultiLayerGAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers==1:
            self.layers.append(GATConv(feature_size, output_size, flow="target_to_source"))
        else:
            self.layers.append(GATConv(feature_size, hidden_size, flow="target_to_source"))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_size, hidden_size, flow="target_to_source"))
            self.layers.append(GATConv(hidden_size, output_size, flow="target_to_source"))

    def forward(self, x, edge_indices):
        for i, edge_index in enumerate(edge_indices):
            new_edge_index = torch.vstack((edge_index[1,:],edge_index[0,:]))
            x = self.layers[i](x, new_edge_index)
            x = torch.relu(x)
            if i != len(edge_indices) - 1:  # Not apply dropout for the last layer
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class MultiLayerGraphSAGE(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(MultiLayerGraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers==1:
            self.layers.append(SAGEConv(feature_size, output_size, flow="target_to_source"))
        else:
            self.layers.append(SAGEConv(feature_size, hidden_size, flow="target_to_source"))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_size, hidden_size, flow="target_to_source"))
            self.layers.append(SAGEConv(hidden_size, output_size, flow="target_to_source"))

    def forward(self, x, edge_indices):
        for i, edge_index in enumerate(edge_indices):
            new_edge_index = torch.vstack((edge_index[1,:],edge_index[0,:]))
            x = self.layers[i](x, new_edge_index)
            x = torch.relu(x)
            if i != len(edge_indices) - 1:  # Not apply dropout for the last layer
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class MultiLayerTransformer(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(MultiLayerTransformer, self).__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers==1:
            self.layers.append(TransformerConv(feature_size, output_size, flow="target_to_source"))
        else:
            self.layers.append(TransformerConv(feature_size, hidden_size, flow="target_to_source"))
            for _ in range(num_layers - 2):
                self.layers.append(TransformerConv(hidden_size, hidden_size, flow="target_to_source"))
            self.layers.append(TransformerConv(hidden_size, output_size, flow="target_to_source"))

    def forward(self, x, edge_indices):
        for i, edge_index in enumerate(edge_indices):
            new_edge_index = torch.vstack((edge_index[1,:],edge_index[0,:]))
            x = self.layers[i](x, new_edge_index)
            x = torch.relu(x)
            if i != len(edge_indices) - 1:  # Not apply dropout for the last layer
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class MultiLayerGIN(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(MultiLayerGIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers==1:
            nn = Seq(Lin(feature_size, output_size))
            self.layers.append(GINConv(nn))
        else:
            nn = Seq(Lin(feature_size, hidden_size))
            self.layers.append(GINConv(nn))
            for _ in range(num_layers - 2):
                nn = Seq(Lin(hidden_size, hidden_size))
                self.layers.append(GINConv(nn))
            nn = Seq(Lin(hidden_size, output_size))
            self.layers.append(GINConv(nn))

    def forward(self, x, edge_indices):
        for i, edge_index in enumerate(edge_indices):
            new_edge_index = torch.vstack((edge_index[1,:],edge_index[0,:]))
            x = self.layers[i](x, new_edge_index)
            x = torch.relu(x)
            if i != len(edge_indices) - 1:  # Not apply dropout for the last layer
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class MultiLayerRGCN(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_relations, num_layers):
        super(MultiLayerRGCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers==1:
            self.layers.append(RGCNConv(feature_size, output_size, num_relations=num_relations))
        else:
            self.layers.append(RGCNConv(feature_size, hidden_size, num_relations=num_relations))
            for _ in range(num_layers - 2):
                self.layers.append(RGCNConv(hidden_size, hidden_size, num_relations=num_relations))
            self.layers.append(RGCNConv(hidden_size, output_size, num_relations=num_relations))

    def forward(self, x, edge_indices, edge_types):
        for i, (edge_index, edge_type) in enumerate(zip(edge_indices, edge_types)):
            x = self.layers[i](x, edge_index, edge_type)
            x = torch.relu(x)
            if i != len(edge_indices) - 1:  # Not apply relu and dropout for the last layer
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        # if isinstance(in_channels, int):
        #     self.lin_src = Linear(in_channels, heads * out_channels,
        #                           bias=False, weight_initializer='glorot')
        #     self.lin_dst = self.lin_src
        # else:
        #     self.lin_src = Linear(in_channels[0], heads * out_channels, False,
        #                           weight_initializer='glorot')
        #     self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
        #                           weight_initializer='glorot')

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src


        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        # if bias and concat:
        #     self.bias = Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self._alpha = None
        self.attentions = None

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.lin_src.reset_parameters()
    #     self.lin_dst.reset_parameters()
    #     glorot(self.att_src)
    #     glorot(self.att_dst)
    #     # zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention = None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention


        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        #alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
        
class STAGATE(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super(STAGATE, self).__init__()

        in_dim, num_hidden, out_dim = feature_size, hidden_size, output_size
        
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

#     def forward(self, features, edge_index):
        
#         h1 = F.elu(self.conv1(features, edge_index))
#         h2 = self.conv2(h1, edge_index, attention=False)
    
#         self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
#         self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
#         self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
#         self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
#         h3 = F.elu(self.conv3(h2, edge_index, attention=True,
#                               tied_attention=self.conv1.attentions))
#         h4 = self.conv4(h3, edge_index, attention=False)

# #         return h2, h4  # F.log_softmax(x, dim=-1)
#         return h4

    def forward(self, features, edge_indices):
        
        assert len(edge_indices) == 2
        
        h1 = F.elu(self.conv1(features, edge_indices[0]))
        h2 = self.conv2(h1, edge_indices[1], attention=False)
    
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        
        h3 = F.elu(self.conv3(h2, edge_indices[1], attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_indices[0], attention=False)

        return h4

    