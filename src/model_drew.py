"""
Simplified DRew (Dynamically Rewired Message Passing with Delay) implementation
Adapted from https://github.com/BenGutteridge/DRew/
Core innovation: k-hop aggregations can use node embeddings from previous layers (delay mechanism)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ReLU, BatchNorm1d, Sequential
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import numpy as np


def create_mlp(in_channels, out_channels, device, batch_norm=True, final_activation=True):
    """Create a 2-layer MLP with optional batch norm and activation"""
    if batch_norm:
        if final_activation:
            mlp = Sequential(
                Linear(in_channels, out_channels),
                BatchNorm1d(out_channels),
                ReLU(),
                Linear(out_channels, out_channels),
                BatchNorm1d(out_channels),
                ReLU()
            )
        else:
            mlp = Sequential(
                Linear(in_channels, out_channels),
                BatchNorm1d(out_channels), 
                ReLU(),
                Linear(out_channels, out_channels)
            )
    else:
        if final_activation:
            mlp = Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels),
                ReLU()
            )
        else:
            mlp = Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels)
            )
    return mlp.to(device)


class DrewLayer(nn.Module):
    """
    Simplified DRew layer implementing delay mechanism for k-hop aggregations
    """
    def __init__(
        self, 
        layer_idx,
        in_channels, 
        out_channels, 
        nu=1,
        max_distance=5,
        eps=0.0,
        batch_norm=True,
        device=None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nu = nu
        self.max_distance = max_distance
        self.eps = eps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MLPs for different hop distances
        self.self_mlp = create_mlp(in_channels, in_channels, self.device, batch_norm, final_activation=False)
        self.neighbor_mlp = create_mlp(in_channels, in_channels, self.device, batch_norm, final_activation=False)
        
        # Different MLPs for different k-hop distances (k >= 2)
        self.khop_mlps = ModuleList([
            create_mlp(in_channels, in_channels, self.device, batch_norm, final_activation=False)
            for _ in range(max_distance - 1)  # k=2,3,...,max_distance
        ])
        
        # Final transformation MLP
        self.final_mlp = create_mlp(in_channels, out_channels, self.device, batch_norm, final_activation=True)
        
        # Learnable weights for combining different hop aggregations
        self.hop_weights = nn.Parameter(torch.randn(max_distance + 1))  # +1 for k=0 (self)
        
    def compute_khop_edges(self, edge_index, num_nodes, max_k):
        """
        Compute k-hop edges up to max_k using BFS
        Returns dict mapping k -> edge_index for k-hop neighbors
        """
        device = edge_index.device
        khop_edges = {}
        
        # Start with 1-hop (direct edges)
        current_edges = edge_index
        visited = set()
        
        for k in range(1, max_k + 1):
            if current_edges.size(1) == 0:
                break
                
            # Store k-hop edges (remove self-loops and already visited)
            mask = current_edges[0] != current_edges[1]  # Remove self-loops
            valid_edges = current_edges[:, mask]
            
            # Remove already seen edges
            edge_tuples = set(zip(valid_edges[0].cpu().numpy(), valid_edges[1].cpu().numpy()))
            new_edges = edge_tuples - visited
            visited.update(new_edges)
            
            if new_edges:
                new_edge_tensor = torch.tensor(list(new_edges), device=device).t()
                khop_edges[k] = new_edge_tensor
            
            if k < max_k:
                # Compute (k+1)-hop edges by extending current k-hop edges
                # This is a simplified version - full implementation would use proper BFS
                if k == 1:
                    # For k=2, we compose 1-hop edges
                    adj_matrix = torch.sparse_coo_tensor(
                        edge_index, 
                        torch.ones(edge_index.size(1), device=device),
                        (num_nodes, num_nodes)
                    )
                    adj2 = torch.sparse.mm(adj_matrix, adj_matrix)
                    current_edges = adj2.coalesce().indices()
                else:
                    # For higher k, this becomes computationally expensive
                    # In practice, DRew precomputes these using shortest path algorithms
                    break
        
        return khop_edges
    
    def forward(self, node_embeddings_history, edge_index, batch):
        """
        Forward pass with delay mechanism
        
        Args:
            node_embeddings_history: List of node embeddings from all previous layers [layer_0, layer_1, ..., layer_t]
            edge_index: Edge indices
            batch: Batch indices
        """
        current_embeddings = node_embeddings_history[self.layer_idx]
        num_nodes = current_embeddings.size(0)
        
        # Self-loop contribution (k=0)
        self_contrib = self.self_mlp(current_embeddings)
        
        # Compute k-hop edges
        khop_edges = self.compute_khop_edges(edge_index, num_nodes, self.max_distance)
        
        # Aggregate contributions from different hops with delay
        hop_contribs = [self_contrib]  # k=0
        
        # 1-hop neighbors (k=1) - no delay
        if 1 in khop_edges:
            edges_1hop = khop_edges[1]
            neighbor_emb = self.neighbor_mlp(current_embeddings)
            
            # Aggregate 1-hop neighbors
            aggr_1hop = scatter_add(
                neighbor_emb[edges_1hop[1]], 
                edges_1hop[0], 
                dim=0, 
                dim_size=num_nodes
            )
            hop_contribs.append(aggr_1hop)
        else:
            hop_contribs.append(torch.zeros_like(self_contrib))
        
        # k-hop neighbors (k >= 2) with delay mechanism
        for k in range(2, self.max_distance + 1):
            if k in khop_edges:
                # Delay mechanism: use embeddings from layer (t - delay)
                delay = max(0, k - self.nu)
                delayed_layer_idx = max(0, self.layer_idx - delay)
                delayed_embeddings = node_embeddings_history[delayed_layer_idx]
                
                # Apply k-hop specific MLP
                mlp_idx = k - 2  # k=2 -> index 0, k=3 -> index 1, etc.
                if mlp_idx < len(self.khop_mlps):
                    khop_emb = self.khop_mlps[mlp_idx](delayed_embeddings)
                    
                    # Aggregate k-hop neighbors
                    edges_khop = khop_edges[k]
                    aggr_khop = scatter_add(
                        khop_emb[edges_khop[1]], 
                        edges_khop[0], 
                        dim=0, 
                        dim_size=num_nodes
                    )
                    hop_contribs.append(aggr_khop)
                else:
                    hop_contribs.append(torch.zeros_like(self_contrib))
            else:
                hop_contribs.append(torch.zeros_like(self_contrib))
        
        # Weighted combination of hop contributions
        hop_stack = torch.stack(hop_contribs, dim=0)  # [num_hops, num_nodes, features]
        weights = F.softmax(self.hop_weights, dim=0)
        
        # Weighted sum across hops
        combined = (hop_stack * weights.view(-1, 1, 1)).sum(dim=0)
        
        # GIN-style update: (1 + eps) * self + aggregated
        gin_update = (1 + self.eps) * current_embeddings + combined
        
        # Final MLP transformation
        output = self.final_mlp(gin_update)
        
        return output


class DRew(nn.Module):
    """
    Simplified DRew model for graph classification
    """
    def __init__(
        self,
        in_channels,
        hidden_channels, 
        out_channels,
        num_layers=2,
        nu=1,
        max_distance=5,
        eps=0.0,
        dropout=0.5,
        batch_norm=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.nu = nu
        self.max_distance = max_distance
        self.dropout = dropout
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input projection
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # DRew layers
        self.layers = ModuleList()
        for i in range(num_layers):
            layer = DrewLayer(
                layer_idx=i,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                nu=nu,
                max_distance=max_distance,
                eps=eps,
                batch_norm=batch_norm,
                device=device
            )
            self.layers.append(layer)
        
        # Output projection  
        self.output_proj = Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch, edge_attr=None):
        # Initial projection
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store embeddings from all layers for delay mechanism
        node_embeddings_history = [x]
        
        # Apply DRew layers
        for layer in self.layers:
            x = layer(node_embeddings_history, edge_index, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)
            node_embeddings_history.append(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        
        # Final output projection
        x = self.output_proj(x)
        
        return x 