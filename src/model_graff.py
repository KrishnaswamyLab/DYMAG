"""
Modified From https://github.com/realfolkcode/GRAFF/blob/main/GRAFF_Tutorial_PyG.ipynb
"""
import os
import torch


import time

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
import torch.nn.utils.parametrize as parametrize
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, homophily

from torch_geometric.datasets import WebKB, Planetoid


from torch_geometric.nn import global_mean_pool


class GRAFF(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, self_loops=True, step_size=1.):
        super().__init__()
        self.step_size = step_size

        # Encoder
        self.enc = torch.nn.Linear(in_channels, hidden_channels, bias=False)

        # Initialize the linear layers
        self.ext_lin = External(hidden_channels)
        self.pair_lin = Pairwise(hidden_channels)
        self.source_lin = Source()

        # Initialize the GRAFF layer
        self.conv = GRAFFConv(self.ext_lin, self.pair_lin, self.source_lin, self_loops=self_loops)

        # Decoder
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.num_layers = num_layers
        self.reset_parameters()
    
    def reset_parameters(self):
        self.enc.reset_parameters()
        self.ext_lin.reset_parameters()
        self.pair_lin.reset_parameters()
        self.source_lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, batch):
        
        # Apply the encoder
        x = self.enc(x)
        # Copy the initial features
        x0 = x.clone()

        # This context manager caches the parametrization to reduce redundant calculations
        with parametrize.cached():
            for _ in range(self.num_layers):
                x = x + self.step_size * F.relu(self.conv(x, edge_index, x0))

        # Apply the decoder
        x = global_mean_pool(x, batch)
        return self.lin(x)

     


class GRAFFConv(MessagePassing):
    def __init__(self, ext_lin, pair_lin, source_lin, self_loops=True):
        super().__init__(aggr='add')
        self.ext_lin = ext_lin
        self.pair_lin = pair_lin
        self.source_lin = source_lin
        self.self_loops = self_loops

    def forward(self, x, edge_index, x0):
        # (Optionally) Add self-loops to the adjacency matrix.
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix.
        out = self.pair_lin(x)

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages.
        out = self.propagate(edge_index, x=out, norm=norm)

        # Add the external and source contributions
        out -= self.ext_lin(x) + self.source_lin(x0)

        return out

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j

class External(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((1, num_features)))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)

    def forward(self, x):
        return x * self.weight
     
class PairwiseParametrization(torch.nn.Module):
    def forward(self, W):
        # Construct a symmetric matrix with zero diagonal
        W0 = W[:, :-2].triu(1)
        W0 = W0 + W0.T

        # Retrieve the `q` and `r` vectors from the last two columns
        q = W[:, -2]
        r = W[:, -1]
        # Construct the main diagonal
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r) 

        return W0 + w_diag


class Pairwise(torch.nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        # Pay attention to the dimensions
        self.lin = torch.nn.Linear(num_hidden + 2, num_hidden, bias=False)
        # Add parametrization
        parametrize.register_parametrization(self.lin, "weight", PairwiseParametrization(), unsafe=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, x):
        return self.lin(x)
     
class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
    
    def forward(self, x):
        return x * self.weight

