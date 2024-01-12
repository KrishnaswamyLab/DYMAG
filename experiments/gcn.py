import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # Pooling to create graph-level representation
        x = self.fc(x)  # Final classification layer
        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()