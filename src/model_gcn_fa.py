"""
Modified from https://github.com/jctops/understanding-oversquashing/blob/main/gdl/src/gdl/models/gcn_fa.py
"""
from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset


class GCN_FA(torch.nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        num_layers: int = 2, 
        dropout: float = 0.5
    ):
        super(GCN_FA, self).__init__()

        # Build layer dimensions
        if num_layers == 1:
            num_features = [in_channels, out_channels]
        else:
            num_features = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        
        layers = []
        # GCN layers (all but the last)
        for in_features, out_features in zip(num_features[:-2], num_features[1:-1]):
            layers.append(GCNConv(in_features, out_features))
        
        # Here's the +FA addition - Linear layer instead of final GCN
        self.lin = Linear(num_features[-2], num_features[-1])

        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, batch, edge_attr=None):
        # GCN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight=edge_attr)
            x = self.act_fn(x)
            x = self.dropout(x)
            
        # Efficient FA operation: sum all node features and broadcast to all nodes
        # Equivalent to torch.matmul(torch.ones(x.shape[0], x.shape[0]), x) but much faster
        x_sum = x.sum(dim=0, keepdim=True)  # Sum all node features
        x = x_sum.expand_as(x)  # Broadcast to all nodes
        
        # Pool to graph level for graph classification
        x = global_mean_pool(x, batch)
        
        # Final linear layer
        return self.lin(x)