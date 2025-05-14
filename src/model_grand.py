import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GRAND(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3, dropout=0.5, dropnode_rate=0.5):
        super().__init__()
        self.K = K
        self.dropout = dropout
        self.dropnode_rate = dropnode_rate
        self.mlp = MLP(in_channels, hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def drop_node(self, x):
        drop_mask = torch.rand(x.size(0), device=x.device) >= self.dropnode_rate
        x = x * drop_mask.unsqueeze(1)
        return x

    def feature_propagation(self, x, edge_index, num_nodes):
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1)
        deg_inv = 1. / deg[row]
        norm = deg_inv
        out = x.clone()
        for _ in range(self.K):
            out = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), x[col] * norm.unsqueeze(1))
            x = out
        return x

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.training:
            x = self.drop_node(x)

        x = self.feature_propagation(x, edge_index, x.size(0))
        x = self.mlp(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)