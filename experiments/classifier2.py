"""
Classifier model using the PDE's for graph classification.
Rewritten to add batchnorm and skip connections.

Xingzhi Sun
May 2023

"""
from torch import nn
import torch.nn.functional as F
from .cheby_poly_layer import ChebyPolyLayer
from .pde_layers import get_cheby_coefs_heat, get_cheby_coefs_wave
from torch_geometric.nn import global_mean_pool

class HeatBlock(nn.Module):
    """
    Wraps the Heat equation layer with flattening, batch_normalization and skip connections.
    """

    def __init__(
            self, 
            coefs,
            n_input_ts, 
            n_hidden,
            batch_norm=False,
        ):
        """_summary_

        Args:
            coefs (_type_): _description_
            device (_type_): _description_
            n_input_ts (_type_): input size times number of time points. 
            n_hidden (_type_): hidden size.
            p_dropout (float, optional): _description_. Defaults to 0.5.
            batch_norm (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.batch_norm = batch_norm
        self.n_input_ts = n_input_ts
        self.n_hidden = n_hidden
        self.pde_layer = ChebyPolyLayer(coefs)
        self.lin_layer = nn.Linear(n_input_ts, n_hidden)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(n_input_ts)
            self.bn2 = nn.BatchNorm1d(n_hidden)
        
    def forward(self, x, edge_index, edge_weight):
        """_summary_

        Args:
            x (_type_): (n, n_input)
            edge_index (_type_): _description_
            edge_weight (_type_): _description_

        Returns:
            _type_: (n, n_hidden)
        """
        x = self.pde_layer(edge_index, edge_weight, x)
        x = x.permute(1, 0, 2).flatten(1, 2)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.lin_layer(x)
        if self.batch_norm:
            x = self.bn2(x)
        return x
    
class WaveBlock(nn.Module):
    """
    Wraps the Wave equation layer with flattening, batch_normalization and skip connections.
    """

    def __init__(
            self, 
            coefsx, 
            coefsy, 
            n_input_ts, 
            n_hidden,
            batch_norm=False,
        ):
        """_summary_

        Args:
            coefsx (_type_): _description_
            coefsy (_type_): _description_
            device (_type_): _description_
            n_hidden (_type_): output shape of the flattened wave layer.
            batch_norm (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.batch_norm = batch_norm
        self.n_hidden = n_hidden
        self.n_input_ts = n_input_ts
        self.pde_layerx = ChebyPolyLayer(coefsx)
        self.pde_layery = ChebyPolyLayer(coefsy)
        self.lin_layer = nn.Linear(n_input_ts, n_hidden)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(n_input_ts)
            self.bn2 = nn.BatchNorm1d(n_hidden)
        
    def forward_x_y(self, x, y, edge_index, edge_weight):
        z = self.pde_layerx(edge_index, edge_weight, x) + self.pde_layery(edge_index, edge_weight, y)
        z = z.permute(1, 0, 2).flatten(1, 2)
        if self.batch_norm:
            z = self.bn1(z)
        z = F.relu(z)
        z = self.lin_layer(z)
        if self.batch_norm:
            z = self.bn2(z)
        return z
    
    def forward(self, x, edge_index, edge_weight):
        """
        TODO For the moment, we use x for both initial position and speed!
        """
        return self.forward_x_y(x, x, edge_index, edge_weight)

class LinearBlock(nn.Module):
    def __init__(
            self,
            n_hidden,
            batch_norm=False,
            skip_conn=False
        ):
        super().__init__()
        self.batch_norm = batch_norm
        self.skip_conn = skip_conn
        self.n_hidden = n_hidden
        self.lin_layer = nn.Linear(n_hidden, n_hidden)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(n_hidden)
        
    def forward(self, x):
        prev_x = x if self.skip_conn else 0.
        x = self.lin_layer(x)
        if self.batch_norm:
            x = self.bn(x)
        x += prev_x
        return x

class PDEClassifier(nn.Module):
    """
    """

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
            num_lin_layers_between_pde=None,  ## DEPRECATED. HARD-CODED to 1.
            num_lin_layers_after_pde=1, 
            p_dropout=0.5, 
            skip_conn=False,
            batch_norm=False,
        ):
        """_summary_

        Args:
            pde (_type_): _description_
            ts (_type_): _description_
            n_input (_type_): _description_
            n_hidden (_type_): _description_
            n_output (_type_): _description_
            device (_type_): _description_
            degree (int, optional): _description_. Defaults to 100.
            c (_type_, optional): _description_. Defaults to 1..
            N (int, optional): _description_. Defaults to 1000.
            num_layers (int, optional): _description_. Defaults to 2.
            num_lin_layers_between_pde (int, optional): _description_. Defaults to 1.
            p_dropout (float, optional): _description_. Defaults to 0.5.
            skip_conn (bool, optional): _description_. Defaults to False.
            batch_norm (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.p_dropout = p_dropout
        self.batch_norm = batch_norm
        self.skip_conn = skip_conn
        self.n_input = n_input
        self.n_output = n_output
        self.device = device
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        # self.num_lin_layers_between_pde = num_lin_layers_between_pde
        if num_lin_layers_between_pde is not None:
            print('WARNING: num_lin_layers_between_pde is DEPRECATED. HARD-CODED to 1.')
        self.num_lin_layers_after_pde = num_lin_layers_after_pde
        self.pde = pde
        assert pde in ['heat', 'wave'], 'Invalid PDE type!'
        if pde == 'heat':
            self.coefs = get_cheby_coefs_heat(ts, degree, c=c, N=N, device=device)
        elif pde == 'wave':
            self.coefsx, self.coefsy = get_cheby_coefs_wave(ts, degree, c=c, N=N, device=device)
        self.pde_layers = nn.ModuleList()
        self.lin_layers_between_pde = nn.ModuleList()
        if pde == 'heat':
            self.pde_layers.append(HeatBlock(self.coefs, len(ts) * n_input, n_hidden, self.batch_norm))
        elif pde == 'wave':
            self.pde_layers.append(WaveBlock(self.coefsx, self.coefsy, len(ts) * n_input, n_hidden, self.batch_norm))

        for i in range(1, num_layers):
            if pde == 'heat':
                self.pde_layers.append(HeatBlock(self.coefs, len(ts) * n_hidden, n_hidden, self.batch_norm))
            elif pde == 'wave':
                self.pde_layers.append(WaveBlock(self.coefsx, self.coefsy, len(ts) * n_hidden, n_hidden, self.batch_norm))
        
        self.lin_layers_after_pde = nn.ModuleList()
        for _ in range(num_lin_layers_after_pde):
            self.lin_layers_after_pde.append(LinearBlock(n_hidden, self.batch_norm, self.skip_conn))

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        for i in range(self.num_layers):
            x = self.pde_layers[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        
        x = global_mean_pool(x, batch)

        for lin_layer in self.lin_layers_after_pde:
            x = lin_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        self.apply(weight_reset)

def weight_reset(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    elif isinstance(m, nn.BatchNorm1d):
        m.reset_running_stats()
        if m.affine:
            m.reset_parameters()
