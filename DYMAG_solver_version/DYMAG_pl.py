import os
import sys
import torch
import pytorch_lightning as pl
from aggregators import KHopSumAggregator, GraphMomentAggregator
from PDE_layer import PDE_layer

class DYMAG_pl(pl.LightningModule):
    def __init__(self,
                 input_feature_dim, 
                 output_dim,
                 dynamics='sprott',
                 K=3, 
                 M=4, 
                 S=4, 
                 num_layers=2, 
                 num_lin_layers_after_pde=2, 
                 learning_rate=1e-3,
                 custom_device='cpu'):
        super(DYMAG_pl, self).__init__()
        
        self.save_hyperparameters()
        self.input_feature_dim = input_feature_dim
        self.output_dim = output_dim 
        self.K = K
        self.M = M
        self.S = S
        self.num_layers = num_layers
        self.num_lin_layers_after_pde = num_lin_layers_after_pde
        self.custom_device = custom_device
        self.learning_rate = learning_rate
        self.dynamics = dynamics

        self.pde_layer = PDE_layer(dynamics=dynamics)
        self.k_hop_sum_aggregator = KHopSumAggregator(self.K, self.M)
        self.graph_moment_aggregator = GraphMomentAggregator(self.S)

        self.time_points = self.pde_layer.output_times
        self.aggregated_size = self.S * self.K * self.M * self.input_feature_dim * len(self.time_points)

        self.lin_layers = torch.nn.ModuleList()
        print('input size is', self.aggregated_size)
        layer_size_list = [self.aggregated_size, 64, 48, 32]
        for i in range(len(layer_size_list) - 1):
            self.lin_layers.append(torch.nn.Linear(layer_size_list[i], layer_size_list[i+1]))
        self.classifier = torch.nn.Linear(layer_size_list[-1], output_dim)

        self.nonlin = torch.nn.LeakyReLU()

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
        return x

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def training_step(self, batch, batch_idx):
        out = self.forward(batch.x, batch.edge_index, batch.batch)
        loss = torch.nn.functional.mse_loss(out, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch.x, batch.edge_index, batch.batch)
        val_loss = torch.nn.functional.mse_loss(out, batch.y)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        out = self.forward(batch.x, batch.edge_index, batch.batch)
        test_loss = torch.nn.functional.mse_loss(out, batch.y)
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == '__main__':
    # Test the model
    num_nodes = 10
    num_features = 5
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    batch_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    model = DYMAG_pl(num_features, 5, heat_derivative_func)
    print(model(x, edge_index, batch_index).shape)
    print(model)
