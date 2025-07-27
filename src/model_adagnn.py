"""
PyTorch Geometric native implementation of AdaGNN
Modified from https://github.com/yushundong/AdaGNN/tree/main
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
from typing import Optional


class AdaGNNConv(MessagePassing):
    """
    PyG implementation of AdaGNN layer using message passing
    """
    def __init__(self, in_features, out_features, bias=True, has_weight=True):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        
        self.in_features = in_features
        self.out_features = out_features
        self.has_weight = has_weight
        
        # Learnable diagonal matrix (always present)
        self.learnable_diag = Parameter(torch.FloatTensor(in_features))
        
        # Weight matrix (only for layers with weights)
        if has_weight:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter('weight', None)
            
        # Bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features if has_weight else in_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.has_weight:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            torch.nn.init.normal_(self.learnable_diag, mean=0, std=0.01)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            torch.nn.init.normal_(self.learnable_diag, mean=0, std=0)
            if self.bias is not None:
                self.bias.data.zero_()
    
    def forward(self, x, edge_index):
        """
        Forward pass implementing AdaGNN's logic exactly:
        1. e1 = L_sym * x  (where L_sym = I - D^(-1/2) A D^(-1/2))
        2. e2 = e1 * diagonal_matrix  
        3. output = x - e2
        """
        # Step 1: Compute L_sym * x = x - D^(-1/2) A D^(-1/2) x
        symmetric_mp = self.propagate(edge_index, x=x)  # D^(-1/2) A D^(-1/2) x
        e1 = x - symmetric_mp  # This is L_sym * x
        
        # Step 2: Apply learnable diagonal matrix
        alpha = torch.diag(self.learnable_diag)
        if self.has_weight:
            # For Adagnn_with_weight: add identity to diagonal
            alpha = alpha + torch.eye(self.in_features, device=x.device)
        
        e2 = torch.mm(e1, alpha)  # e1 * diagonal_matrix
        
        # Step 3: Compute x - e2 (this matches original AdaGNN exactly)
        e4 = x - e2
        
        # Step 4: Apply weight matrix if present
        if self.has_weight:
            output = torch.mm(e4, self.weight)
        else:
            output = e4
            
        # Add bias
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def message(self, x_j, norm):
        """Apply symmetric normalization during message passing"""
        return norm.view(-1, 1) * x_j
    
    def propagate(self, edge_index, x):
        """Compute symmetric normalized message passing"""
        # Compute degree
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Apply D^(-1/2) to source nodes  
        x_norm = deg_inv_sqrt.view(-1, 1) * x
        
        # Compute normalization for target nodes
        norm = deg_inv_sqrt[row]
        
        # Propagate messages with symmetric normalization
        out = super().propagate(edge_index, x=x_norm, norm=norm)
        
        return out


class AdaGNN(nn.Module):
    """
    PyG-native AdaGNN model
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        
        # First layer (with weight)
        self.first_layer = AdaGNNConv(in_channels, hidden_channels, has_weight=True)
        
        # Hidden layers (without weight)  
        self.hidden_layers = nn.ModuleList([
            AdaGNNConv(hidden_channels, hidden_channels, has_weight=False)
            for _ in range(num_layers - 2)
        ])
        
        # Last layer (with weight)
        self.last_layer = AdaGNNConv(hidden_channels, out_channels, has_weight=True)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass compatible with PyG API
        """
        # First layer with activation and dropout
        x = F.relu(self.first_layer(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
            
        # Last layer
        x = self.last_layer(x, edge_index)
        
        # Handle batching - pool to graph level if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        return F.log_softmax(x, dim=1) 