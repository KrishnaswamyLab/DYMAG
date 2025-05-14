# src/model_graphormer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool
from torch_geometric.utils import degree

class Graphormer(nn.Module):
    """
    Graphormer-lite: TransformerConv stack + degree-centrality encoding.
    constant edge_attr (all-ones) is expected (edge_dim=1).
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 heads: int = 8,
                 edge_dim: int = 1):
        super().__init__()

        # +1 for degree scalar
        self.input_proj = nn.Linear(in_channels + 1, hidden_channels)

        # first layer
        self.convs = nn.ModuleList([
            TransformerConv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                concat=True,
                edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, edge_attr):
        # centrality encoding (Graphormer Eq-3)  https://arxiv.org/pdf/2106.05234
        deg = degree(edge_index[0], num_nodes=x.size(0)).unsqueeze(1)
        deg = deg / deg.max().clamp(min=1)       # scale 0-1
        x   = self.input_proj(torch.cat([x, deg], dim=1))

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))

        x = global_add_pool(x, batch)
        x = self.readout(x)
        return self.classifier(x)