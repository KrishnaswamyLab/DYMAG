# make a torch class that applys a PDE_layer, then a KHopSumAggregator, then a GraphMomentAggregator, then flattens the output
# before passing it to a classifier
import os
import sys
import torch
from .aggregators import KHopSumAggregator, GraphMomentAggregator
from .PDE_layer import PDE_layer, heat_derivative_func


class DYMAG(torch.nn.Module):
    def __init__(self, num_nodes, num_features, num_outputs = 26, num_layers=2, num_lin_layers_after_pde=2, device='cpu'):
        super(DYMAG, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_lin_layers_after_pde = num_lin_layers_after_pde
        self.device = device

        self.pde_layer = PDE_layer(num_nodes, heat_derivative_func)
        self.k_hop_sum_aggregator = KHopSumAggregator()
        self.graph_moment_aggregator = GraphMomentAggregator()

        self.lin_layers = torch.nn.ModuleList()

        input_size = 26 * 3 * 4 * 4 * 6 # TODO: FIX PARAMETER SPECIFICATION AND INPUT
        print('input size is ' , input_size)
        for i in range(num_lin_layers_after_pde):
            self.lin_layers.append(torch.nn.Linear(input_size, input_size))

        self.classifier = torch.nn.Linear(input_size, num_outputs)

    def forward(self, x, edge_index):
        print('input data has shape ' , x.shape)
        x = self.pde_layer(x, edge_index)
        print('pde output has shape ' , x.shape)
        x = self.k_hop_sum_aggregator(x, edge_index)
        print('k hop sum aggregator has shape ' , x.shape)
        x = self.graph_moment_aggregator(x, edge_index)
        print('graph moment sum aggregator has shape ' , x.shape)

        x = x.view(-1)
        print('flattened output has shape ' , x.shape)
        
        for lin_layer in self.lin_layers:
            x = lin_layer(x)
            x = torch.relu(x)

        x = self.classifier(x)
        return x
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

if __name__ == '__main__':
    # test the model
    num_nodes = 10
    num_features = 6
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    model = DYMAG(num_nodes, num_features, 26, 3, 2, 'cpu')
    print(model(x, edge_index).shape)
    print(model)
    model.reset_parameters()
    print(model)