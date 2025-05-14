import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ModuleList, Dropout
from torch_geometric.nn import GINEConv, GINConv, GPSConv, global_add_pool

class GraphGPS(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers=2, heads=4):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        def create_local_mp(dim):
            return GINConv(
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim)
                )
            )

        self.convs = nn.ModuleList([
            GPSConv(channels=hidden_channels,
                    conv=create_local_mp(hidden_channels),
                    heads=heads)
            for _ in range(num_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return self.classifier(x)