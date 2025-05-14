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

class GRANDPP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3, dropout=0.5, dropnode_rate=0.5, n_views=4, lam_consistency=1.0):
        super().__init__()
        self.K = K
        self.dropout = dropout
        self.dropnode_rate = dropnode_rate
        self.n_views = n_views
        self.lam_consistency = lam_consistency

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
        norm = 1.0 / deg[row]
        for _ in range(self.K):
            out = torch.zeros_like(x)
            out = out.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), x[col] * norm.unsqueeze(1))
            x = out
        return x

    def forward(self, data, train_mode=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)

        if train_mode:
            views = []
            for _ in range(self.n_views):
                x_view = self.drop_node(x)
                x_prop = self.feature_propagation(x_view, edge_index, x.size(0))
                h = self.mlp(x_prop)
                graph_feat = global_mean_pool(h, batch)
                views.append(graph_feat)

            # Supervised loss on first view
            logits = self.classifier(views[0])

            # Consistency loss between views
            probs = [F.softmax(self.classifier(v.detach()), dim=1) for v in views]
            mean_probs = sum(probs) / len(probs)
            loss_cons = sum([(p - mean_probs).pow(2).sum(1).mean() for p in probs]) / len(probs)

            # Entropy regularization
            loss_ent = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(1).mean()

            return logits, self.lam_consistency * loss_cons + loss_ent
        else:
            # Eval mode - single forward pass with feature propagation
            x = self.feature_propagation(x, edge_index, x.size(0))
            x = self.mlp(x)
            x = global_mean_pool(x, batch)
            return self.classifier(x)