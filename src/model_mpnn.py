import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.mlp = Sequential(Linear(2 * in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=1))

class MPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(MPNNLayer(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(MPNNLayer(hidden_channels, hidden_channels))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)