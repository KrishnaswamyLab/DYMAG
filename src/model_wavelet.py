import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import global_mean_pool
from src.wavelet_conv import WaveletConv

class Wavelet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, mean_pool_between_layers=False):
        super().__init__()
        self.in_proj = Linear(in_channels, hidden_channels)
        self.convs = ModuleList()
        self.mlps = ModuleList()
        self.mean_pool_between_layers = mean_pool_between_layers
        reshape = not mean_pool_between_layers
        n_wavelets = 1 if mean_pool_between_layers else 6
        self.convs.append(WaveletConv(reshape=reshape))
        self.mlps.append(Sequential(
            Linear(hidden_channels * n_wavelets, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels)
        ))

        for _ in range(num_layers - 1):
            self.convs.append(WaveletConv(reshape=reshape))
            self.mlps.append(Sequential(
                Linear(hidden_channels * n_wavelets, hidden_channels), 
                ReLU(), 
                Linear(hidden_channels, hidden_channels)
            ))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.in_proj(x)
        for conv, mlp in zip(self.convs, self.mlps):
            x = conv(x, edge_index) # Shape: [num_nodes, features, num_wavelets]
            x = F.relu(x)
            if self.mean_pool_between_layers:
                x = x.mean(dim=-1) # take average over the wavelets, Shape: [num_nodes, features]
            x = mlp(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)
