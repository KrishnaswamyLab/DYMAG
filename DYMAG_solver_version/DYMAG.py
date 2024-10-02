import os
import sys
import torch
from aggregators import KHopSumAggregator, GraphMomentAggregator
from PDE_layer import PDE_layer

# make a torch class that applies a PDE_layer, then a KHopSumAggregator, then a GraphMomentAggregator, then flattens the output
# before passing it to a classifier


class DYMAG(torch.nn.Module):
    def __init__(self,
                 input_feature_dim, 
                 output_dim,
                 dynamics='sprott',
                 n_largest_graph=10,
                 K = 3, 
                 M = 4, 
                 S = 4, 
                 num_layers=2, 
                 num_lin_layers_after_pde=2, 
                 device='cpu',
                 ):
        super(DYMAG, self).__init__()
        
        self.input_feature_dim = input_feature_dim
        self.output_dim = output_dim 
        self.K = K
        self.M = M
        self.S = S
        self.dynamics = dynamics

        self.num_layers = num_layers
        self.num_lin_layers_after_pde = num_lin_layers_after_pde
        self.device = device

        self.pde_layer = PDE_layer(dynamics=dynamics, n_largest_graph=n_largest_graph)
        self.k_hop_sum_aggregator = KHopSumAggregator(self.K, self.M)
        self.graph_moment_aggregator = GraphMomentAggregator(self.S)

        self.time_points = self.pde_layer.output_times
        self.aggregated_size = self.S * self.K * self.M * self.input_feature_dim * len(self.time_points)

        self.lin_layers = torch.nn.ModuleList()
        
        print('input size is', self.aggregated_size)
        layer_size_list = [self.aggregated_size, 64, 48,  32]
        for i in range(len(layer_size_list) - 1):
            self.lin_layers.append(torch.nn.Linear(layer_size_list[i], layer_size_list[i+1]))
        self.classifier = torch.nn.Linear(layer_size_list[-1], output_dim)

        self.nonlin = torch.nn.LeakyReLU()
        self.outnonlin = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch_index):
        x = self.pde_layer(x, edge_index, batch_index)
        x = self.k_hop_sum_aggregator(x, edge_index)
        x = self.graph_moment_aggregator(x, batch_index)

        # keep first axis but flatten all rest
        x = x.view(x.size(0), -1)

        for lin_layer in self.lin_layers:
            x = lin_layer(x)
            x = self.nonlin(x)

        x = self.classifier(x)
        # try without sigmoid
        return x
        #return self.outnonlin(x)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


if __name__ == '__main__':
    # test the model
    num_nodes = 10
    num_features = 100
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    batch_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    model = DYMAG(num_features, 1)
    print(model(x, edge_index, batch_index).shape)
    print(model)
    # get the number of trainable parameters for the model
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    import pdb; pdb.set_trace()
    
