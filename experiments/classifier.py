"""
[DEPRECATED. Use experiments/classifier2.py instead]
Classifier model using the PDE's for graph classification.
for time, use time convolution layer (set n_time_conv = 1 if don't want time convolution)

***
April 2023

"""

from torch import nn
import torch.nn.functional as F
from .cheby_poly_layer import ChebyPolyLayer
from .pde_layers import get_cheby_coefs_heat, get_cheby_coefs_wave
from torch_geometric.nn import global_mean_pool

class GraphClassifier(nn.Module):
    def __init__(self, pde, ts, n_input, n_hidden, n_output, device, degree=100, c=1., N=1000):
        """_summary_
        TODO add dropout.

        Args:
            pde (string): type of pde. ('heat', 'wave')
            ts (tensor): sample time points, shape (T, ). if using convolution, preferrably ordered and equal intervals.
            n_input (int): number of input features
            n_hidden (int): number of hidden dimensions
            n_output (int): number of output classes
            n_time_conv (int): number of time convolutional layers
            n_time_pool (int): number of time pooling layers
            device (torch device): device 'cpu' or 'cuda'
            degree (int, optional): degree of chebyshev polynomial approximation. Defaults to 100.
            c (float): wave speed, optional, default=1.0
                the wave eqn is d^2u/dt^2 = - c^2 L u.
            c (float, optional): the coefficient of the heat equation. Defaults to 1.
                the heat eqn is du/dt = - c^2 L u. we use c^2 to make sure the exponential does not explode.
            N (int, optional): the number of points to be used in the integration. Defaults to 1000.

        Raises:
            ValueError: _description_
        """
        super(GraphClassifier, self).__init__()
        self.pde = pde
        self.ts = ts
        self.n_nidden = n_hidden
        if pde == 'heat':
            self.coefs = get_cheby_coefs_heat(ts, degree, c=c, N=N, device=device)
            self.pde_layer1 = ChebyPolyLayer(self.coefs)
        elif pde == 'wave':
            self.coefsx, self.coefsy = get_cheby_coefs_wave(ts, degree, c=c, N=N, device=device)
            self.pde_layer1x = ChebyPolyLayer(self.coefsx)
            self.pde_layer1y = ChebyPolyLayer(self.coefsy)

        else: raise ValueError('Invalid PDE type!')
        # self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=n_hidden, kernel_size=n_time_conv)
        # self.pool1 = nn.MaxPool1d(kernel_size=n_time_pool)
        # self.pde_layer2 = ChebyPolyLayer(self.coefs)
        # self.lin2 = nn.Linear(n_hidden, n_hidden)
        # pooled_output_length = int(((len(ts) - n_time_conv) + 1 - n_time_pool) / n_time_pool + 1)
        self.lin1 = nn.Linear(len(ts) * n_input, n_hidden)
        self.lin2 = nn.Linear(len(ts) * n_hidden, n_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            # nn.Softmax(dim=1)
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        if self.pde == 'heat':
            x = self.pde_layer1(edge_index, edge_weight, x)
        elif self.pde == 'wave':
            ## TODO now use x for both initial position and velocity. maybe should use a different velocity?
            x = self.pde_layer1x(edge_index, edge_weight, x) + self.pde_layer1y(edge_index, edge_weight, x)
        x = F.relu(x) # shape (T, n, m)
        x = self.lin1(x.permute(1, 0, 2).flatten(1, 2)) # shape (n, T*m)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        if self.pde == 'heat':
            x = self.pde_layer1(edge_index, edge_weight, x)
        elif self.pde == 'wave':
            ## TODO now use x for both initial position and velocity. maybe should use a different velocity?
            x = self.pde_layer1x(edge_index, edge_weight, x) + self.pde_layer1y(edge_index, edge_weight, x)
        x = F.relu(x)
        x = self.lin2(x.permute(1, 0, 2).flatten(1, 2))
        x = global_mean_pool(x, batch)
        x = F.relu(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.classifier(x)
        return x
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class GraphClassifierTimeConv(nn.Module):
    def __init__(self, pde, ts, n_input, n_hidden, n_output, n_time_conv, n_time_pool, device, degree=100, c=1., N=1000):
        """_summary_
        TODO add batch norm

        Args:
            pde (string): type of pde. ('heat', 'wave')
            ts (tensor): sample time points, shape (T, ). if using convolution, preferrably ordered and equal intervals.
            n_input (int): number of input features
            n_hidden (int): number of hidden dimensions
            n_output (int): number of output classes
            n_time_conv (int): number of time convolutional layers
            n_time_pool (int): number of time pooling layers
            device (torch device): device 'cpu' or 'cuda'
            degree (int, optional): degree of chebyshev polynomial approximation. Defaults to 100.
            c (float): wave speed, optional, default=1.0
                the wave eqn is d^2u/dt^2 = - c^2 L u.
            c (float, optional): the coefficient of the heat equation. Defaults to 1.
                the heat eqn is du/dt = - c^2 L u. we use c^2 to make sure the exponential does not explode.
            N (int, optional): the number of points to be used in the integration. Defaults to 1000.

        Raises:
            ValueError: _description_
        """
        super(GraphClassifierTimeConv, self).__init__()
        self.pde = pde
        self.ts = ts
        if pde == 'heat':
            self.coefs = get_cheby_coefs_heat(ts, degree, c=c, N=N, device=device)
            self.pde_layer1 = ChebyPolyLayer(self.coefs)
        elif pde == 'wave':
            self.coefsx, self.coefsy = get_cheby_coefs_wave(ts, degree, c=c, N=N, device=device)
            self.pde_layer1x = ChebyPolyLayer(self.coefsx)
            self.pde_layer1y = ChebyPolyLayer(self.coefsy)

        else: raise ValueError('Invalid PDE type!')
        self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=n_hidden, kernel_size=n_time_conv)
        self.pool1 = nn.MaxPool1d(kernel_size=n_time_pool)
        # self.pde_layer2 = ChebyPolyLayer(self.coefs)
        # self.lin2 = nn.Linear(n_hidden, n_hidden)
        pooled_output_length = int(((len(ts) - n_time_conv) + 1 - n_time_pool) / n_time_pool + 1)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_output_length * n_hidden, n_output),
            # nn.Softmax(dim=1)
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        if self.pde == 'heat':
            x = self.pde_layer1(edge_index, edge_weight, x)
        elif self.pde == 'wave':
            ## TODO now use x for both initial position and velocity. maybe should use a different velocity?
            x = self.pde_layer1x(edge_index, edge_weight, x) + self.pde_layer1y(edge_index, edge_weight, x)
        x = self.conv1(x.permute(1, 2, 0))
        x = self.pool1(x).permute(2, 0, 1)
        x = F.relu(x)
        # x = self.pde_layer2(edge_index, edge_weight, x)
        # x = self.lin2(x)
        # x = F.relu(x)
        x = global_mean_pool(x, batch)
        # Flatten the pooled tensor
        ## x now has shape (n_pooled, n_graphs, n_hidden), so permute to (n_graphs, n_pooled, n_hidden) and flatten.
        x = x.permute(1, 0, 2).flatten(1, 2)
        x = F.relu(x)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class GraphClassifierCustomizable(nn.Module):
    def __init__(
            self, 
            pde, 
            ts,
            n_input, 
            n_hidden, 
            n_output, 
            device, 
            degree=100, 
            c=1., 
            N=1000, 
            num_layers=2, 
            num_lin_layers_between_pde=1,  
            num_lin_layers_after_pde=1, 
            p_dropout=0.5, 
            skip_conn=False
        ):
        """_summary_
        TODO add dropout.
        TODO add skip connections.

        Args:
            pde (string): type of pde. ('heat', 'wave')
            ts (tensor): sample time points, shape (T, ). if using convolution, preferrably ordered and equal intervals.
            n_input (int): number of input features
            n_hidden (int): number of hidden dimensions
            n_output (int): number of output classes
            n_time_conv (int): number of time convolutional layers
            n_time_pool (int): number of time pooling layers
            device (torch device): device 'cpu' or 'cuda'
            degree (int, optional): degree of chebyshev polynomial approximation. Defaults to 100.
            c (float): wave speed, optional, default=1.0
                the wave eqn is d^2u/dt^2 = - c^2 L u.
            c (float, optional): the coefficient of the heat equation. Defaults to 1.
                the heat eqn is du/dt = - c^2 L u. we use c^2 to make sure the exponential does not explode.
            N (int, optional): the number of points to be used in the integration. Defaults to 1000.

        Raises:
            ValueError: _description_
        """
        super(GraphClassifierCustomizable, self).__init__()
        self.p_dropout = p_dropout
        self.skip_conn = skip_conn
        assert pde in ['heat', 'wave'], 'Invalid PDE type!'
        self.pde = pde
        self.ts = ts
        if pde == 'heat':
            self.coefs = get_cheby_coefs_heat(ts, degree, c=c, N=N, device=device)
        elif pde == 'wave':
            self.coefsx, self.coefsy = get_cheby_coefs_wave(ts, degree, c=c, N=N, device=device)
        self.n_nidden = n_hidden
        self.num_layers = num_layers
        self.num_lin_layers_between_pde = num_lin_layers_between_pde
        self.num_lin_layers_after_pde = num_lin_layers_after_pde
        if pde == 'heat':
            self.pde_layers = nn.ModuleList()
        elif pde == 'wave':
            self.pde_layersx = nn.ModuleList()
            self.pde_layersy = nn.ModuleList()

        self.lin_layers_between_pde = nn.ModuleList()
        for i in range(num_layers):
            if pde == 'heat':
                self.pde_layers.append(ChebyPolyLayer(self.coefs))
            elif pde == 'wave':
                self.pde_layersx.append(ChebyPolyLayer(self.coefsx))
                self.pde_layersy.append(ChebyPolyLayer(self.coefsy))
            lin_layers = []
            lin_layers.append(nn.Linear(len(ts) * n_hidden, n_hidden)) ## the first one is reshaped.
            for _ in range(1, num_lin_layers_between_pde):
                lin_layers.append(nn.Linear(n_hidden, n_hidden))
            if i == 0:
                lin_layers[0] = nn.Linear(len(ts) * n_input, n_hidden) ## input dim is not hidden dim.
            self.lin_layers_between_pde.append(nn.ModuleList(lin_layers))

        self.lin_layers_after_pde = nn.ModuleList()
        for _ in range(num_lin_layers_after_pde):
            self.lin_layers_after_pde.append(nn.Linear(n_hidden, n_hidden))

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        x_prev = 0.
        for i in range(self.num_layers):
            if self.pde == 'heat':
                x = self.pde_layers[i](edge_index, edge_weight, x)
            elif self.pde == 'wave':
                x = self.pde_layersx[i](edge_index, edge_weight, x) + self.pde_layersy[i](edge_index, edge_weight, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            x = x.permute(1, 0, 2).flatten(1, 2) # shape (n, T*m)

            for lin_layer in self.lin_layers_between_pde[i]:
                x = lin_layer(x)
                x += x_prev
                x = F.relu(x)
                x_prev = x if self.skip_conn else 0.
                x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = global_mean_pool(x, batch)
        for lin_layer in self.lin_layers_after_pde:
            x = lin_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.classifier(x)
        return x
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()