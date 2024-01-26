"""
GDE-RF
"""

import torch
from torch import nn

from .cheby_poly_layer import ChebyPolyLayer
from .pde_layers import get_cheby_coefs_heat, get_cheby_coefs_wave

class GDeFeaturizer(nn.Module):
    def __init__(self, pde, ts, device, degree=100, c=1., N=100) -> None:
        if pde == 'heat':
            self.coefs = get_cheby_coefs_heat(ts, degree, c=c, N=N, device=device)
            self.pde_layer = ChebyPolyLayer(self.coefs)
        elif pde == 'wave':
            self.coefsx, self.coefsy = get_cheby_coefs_wave(ts, degree, c=c, N=N, device=device)
            self.pde_layerx = ChebyPolyLayer(self.coefsx)
            self.pde_layery = ChebyPolyLayer(self.coefsy)
        
    def forward_x(self, x, edge_index, edge_weight):
        assert self.pde == 'heat'
        z = self.pde_layer(edge_index, edge_weight, x)
        return z
    
    def forward_xy(self, x, y, edge_index, edge_weight):
        assert self.pde == 'wave'
        z = self.pde_layerx(edge_index, edge_weight, x) + self.pde_layery(edge_index, edge_weight, y)
        z = z.permute(1, 0, 2).flatten(1, 2)
        return z
    
    def forward(self, x, edge_index, edge_weight):
        """
        TODO For the moment, we use x for both initial position and speed!
        """
        if self.pde == 'heat':
            return self.forward_x(x, edge_index, edge_weight)
        elif self.pde == 'wave':
            return self.forward_xy(x, x, edge_index, edge_weight)

