"""
Modified from https://github.com/tk-rusch/GraphCON
"""
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool



class GraphCON(nn.Module):
    def __init__(self, GNNs, dt=1., alpha=1., gamma=1., dropout=None):
        super(GraphCON, self).__init__()
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma
        self.GNNs = GNNs  # list of the individual GNN layers
        self.dropout = dropout

    def forward(self, X0, Y0, edge_index):
        # set initial values of ODEs
        X = X0
        Y = Y0
        # solve ODEs using simple IMEX scheme
        for gnn in self.GNNs:
            Y = Y + self.dt * (torch.relu(gnn(X, edge_index)) - self.alpha * Y - self.gamma * X)
            X = X + self.dt * Y

            if (self.dropout is not None):
                Y = F.dropout(Y, self.dropout, training=self.training)
                X = F.dropout(X, self.dropout, training=self.training)

        return X, Y
    


class GraphCON_model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dt=1., alpha=1., gamma=1., dropout=None):
        super(GraphCON_model, self).__init__()
        self.enc = nn.Linear(in_channels, hidden_channels)
        self.GNNs = nn.ModuleList()
        for _ in range(num_layers):
            self.GNNs.append(GCNConv(hidden_channels, hidden_channels))
        self.graphcon = GraphCON(self.GNNs, dt, alpha, gamma, dropout)
        self.dec = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # compute initial values of ODEs (encode input)
        X0 = self.enc(x)
        Y0 = X0
        # stack GNNs using GraphCON
        X, Y = self.graphcon(X0, Y0, edge_index)
        # decode X state of GraphCON at final time for output nodes
        output = global_mean_pool(X, batch)
        output = self.dec(output)
        return output