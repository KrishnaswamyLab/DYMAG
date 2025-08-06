import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout
from torch_geometric.nn import ChebConv, global_mean_pool


class ChebNet(torch.nn.Module):
    """
    ChebNet model using Chebyshev spectral convolution
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        K=3,  # Order of Chebyshev polynomials
        normalization='sym',  # Laplacian normalization
        bias=True,
        dropout=0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.K = K
        self.normalization = normalization
        self.dropout = dropout
        
        # Create Chebyshev convolution layers
        self.convs = ModuleList()
        
        # First layer
        self.convs.append(ChebConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                K=K,
                normalization=normalization,
                bias=bias
            ))
        
        # Final convolution layer
        if num_layers > 1:
            self.convs.append(ChebConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                K=K,
                normalization=normalization,
                bias=bias
            ))
        
        # Final linear layer for classification
        final_in_channels = hidden_channels if num_layers > 0 else in_channels
        self.lin = Linear(final_in_channels, out_channels)
        
        # Dropout layer
        self.dropout_layer = Dropout(p=dropout)
        
    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for pooling [num_nodes]
            edge_weight: Optional edge weights [num_edges]
        """
        # Apply Chebyshev convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        
        # Final classification layer
        x = self.lin(x)
        
        return x
    
    def reset_parameters(self):
        """Reset all parameters"""
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters() 