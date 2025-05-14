import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4):
        super().__init__()
        self.convs = ModuleList()

        # First layer: input -> hidden
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6, concat=True))
        hidden_dim = hidden_channels * heads  # because concat=True

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_channels, heads=1, dropout=0.6, concat=True))
            hidden_dim = hidden_channels  # now heads=1, concat=True ⇒ output dim = hidden_channels

        # Last GAT layer
        self.convs.append(GATConv(hidden_dim, hidden_channels, heads=1, dropout=0.6, concat=False))

        # Final classifier
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)