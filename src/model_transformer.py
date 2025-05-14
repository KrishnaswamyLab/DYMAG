import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ModuleList
from torch_geometric.nn import TransformerConv, global_add_pool

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers=3, heads=8):
        super().__init__()
        self.convs = ModuleList()

        self.convs.append(
            TransformerConv(in_channels,
                            hidden_channels // heads,
                            heads=heads, concat=True))
        # Add remaining layers
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(hidden_channels,
                                hidden_channels // heads,
                                heads=heads, concat=True))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return self.classifier(x)