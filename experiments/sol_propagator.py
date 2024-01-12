"""
[DEPRECATED]
"""
import torch
from .sol_operator import PDESolutionOperator

class PDESolutionPropogator():
    def __init__(self, laplacian, device, eps=1e-8) -> None:
        self.operator = PDESolutionOperator(laplacian, device, eps)
        self.device = device
        
    def propogateHeat(self, x, t, c=1.):
        """_summary_

        Args:
            x (tensor): shape (n, *, *)
            t (tensor): shape (m)
            c (float): heat capacity, optional, default=1.0
                the heat eqn is du/dt = - c^2 L u. we use c^2 to make sure the exponential does not explode.

        Returns:
            xts (tensor): shape (m, n, *, *)
        """
        c2 = c * c
        heat_sol_eigs = self.operator.getHeatEqnSolOperEigen(c2 * t)
        return torch.einsum('ij,tj,kj,kab->tiab', self.operator.V, heat_sol_eigs, self.operator.V, x)
    
    def propogateWave(self, x, y, t, c=1.):
        """_summary_

        Args:
            x (tensor): shape (n, *, *)
            y (tensor): shape (n, *, *)
            t (tensor): shape (m)
            c (float): wave speed, optional, default=1.0
                the wave eqn is d^2u/dt^2 = - c^2 L u.

        Returns:
            xts (tensor): shape (m, n, *, *)
        """
        wave_sol_eigs1, wave_sol_eigs2 = self.operator.getWaveEqnSolOperEigen(c * t)
        P1x = torch.einsum('ij,tj,kj,kab->tiab', self.operator.V, wave_sol_eigs1, self.operator.V, x)
        P2y = torch.einsum('ij,tj,kj,kab->tiab', self.operator.V, wave_sol_eigs2, self.operator.V, y)
        return P1x + P2y