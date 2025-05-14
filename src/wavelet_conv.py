import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ModuleList, Dropout
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import torch
from typing import Optional, Tuple

class DiffusionConv(MessagePassing):
    def __init__(
            self, 
            normalize="left",
            lazy: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.normalize = normalize
        self.lazy = lazy

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if hasattr(edge_index, 'size') and edge_index.size(0) == 2:
            edge_degree = torch.zeros(x.size(0), device=x.device)
            edge_degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
        else:
            edge_degree = scatter(torch.ones_like(edge_index[0]), edge_index[1],
                        dim=0, dim_size=x.size(0), reduce='sum')
        edge_degree_inv = 1.0 / edge_degree
        edge_degree_inv[edge_degree_inv == float("inf")] = 0

        if self.normalize == "left":
            out = self.propagate(edge_index, x=x, norm=edge_degree_inv)
        elif self.normalize == "right":
            out = edge_degree_inv.view(-1, 1) * x
            out = self.propagate(edge_index, x=out, norm=None)
        elif self.normalize == "symmetric":
            edge_degree_inv_sqrt = edge_degree_inv.sqrt()
            out = edge_degree_inv_sqrt.view(-1, 1) * x
            out = self.propagate(edge_index, x=out, norm=edge_degree_inv_sqrt)
        else:
            raise ValueError(f"Invalid normalization method: {self.normalize}")
        if self.lazy:
            out = 0.5 * (x + out)
        return out
    
    def message(self, x_j: torch.Tensor, norm_i: Optional[torch.Tensor] = None) -> torch.Tensor:
        if norm_i is None:
            out = x_j
        else:
            out = norm_i.view(-1, 1) * x_j
        return out
    
class WaveletConv(MessagePassing):
    def __init__(
            self, 
            normalize="left",
            lazy: bool = True,
            scale_list = None,
            reshape = False,
            **kwargs,
    ):
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.normalize = normalize
        self.lazy = lazy
        self.reshape = reshape
        self.diffusion_conv = DiffusionConv(normalize, lazy, **kwargs)
        if scale_list is None:
            wavelet_matrix = torch.tensor([
                [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=torch.float)
            self.diffusion_levels = 16
        else:
            # ensure that scale list is an increasing list of integers with 0 as the first element
            # ensure that 1 is the second element
            assert all(isinstance(x, int) for x in scale_list)
            assert all(scale_list[i] < scale_list[i+1] for i in range(len(scale_list)-1))
            assert scale_list[0] == 0
            assert scale_list[1] == 1

            self.diffusion_levels = scale_list[-1]
            wavelet_matrix = torch.zeros(len(scale_list), self.diffusion_levels+1, dtype=torch.float)
            for i in range(len(scale_list) - 1):
                wavelet_matrix[i, scale_list[i]] = 1
                wavelet_matrix[i, scale_list[i+1]] = -1
            wavelet_matrix[-1, -1] = 1
        self.register_buffer('wavelet_constructor', wavelet_matrix)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_features = [x]

        for i in range(self.diffusion_levels):
            node_feat = self.diffusion_conv(x=node_features[-1], edge_index=edge_index)
            node_features.append(node_feat)

        diffusion_levels = torch.stack(node_features, dim=0)  # Shape: [levels+1, num_nodes, features]
        wavelet_coeffs = torch.einsum("ij,jkl->kli", self.wavelet_constructor, diffusion_levels) # Shape: [num_nodes, features, num_wavelets]
        
        # Reshape to [num_nodes, num_wavelets * features] if needed
        if self.reshape:
            batch_size, num_features, num_wavelets = wavelet_coeffs.shape
            wavelet_coeffs = wavelet_coeffs.reshape(batch_size, -1)
            
        return wavelet_coeffs