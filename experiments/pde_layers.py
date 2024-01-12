"""
Use Chebyshev polynomials for fast computation of the heat/wave equation solutions.
This file contains the implementations of the HeatLayer and the WaveLayer classes.

Xingzhi Sun
April 2023

written as pytorch_geometric layers.

"""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from .cheby_poly_layer import ChebyLayer, get_cheby_coefs


class HeatLayer(nn.Module):
    """
    [DEPRECATED] See experiments/GraphClassifier for latest usage example.
    The layer for computing the heat equation solution.
    """
    def __init__(self, edge_index, edge_weight, degree=100, c=1., N=1000):
        """
        initiallized the ChebyLayer.

        Args:
            edge_index (tensor): shape (2, E). E is number of edges.
            edge_weight (tensor, optional): shape (E, ).
            degree (int): the degree of the Chebyshev polynomials. (the min power is 0, the max power is degree - 1)
            c (float): heat capacity, optional, default=1.0
                the heat eqn is du/dt = - c^2 L u. we use c^2 to make sure the exponential does not explode.
            N (int, optional): the number of points to be used in the integration. Defaults to 1000.
        """
        super(HeatLayer, self).__init__()
        self.cheby_layer = ChebyLayer(edge_index=edge_index, edge_weight=edge_weight)
        self.degree = degree
        self.N = N
        self.c = c

    def forward(self, x, ts):
        """
        compute the heat equation solution.

        Args:
            ts (tensor): shape (T, ). the sample time points.
            x (tensor): shape (n, *, *). the initial heat.

        Returns:
            tensor: shape (T, n, *, *). the heat equation solution.
        """
        
        coefs_heat = get_cheby_coefs_heat(ts, degree=self.degree, c=self.c, N=self.N)
        yHeat = self.cheby_layer(x, coefs_heat)
        return yHeat

class WaveLayer(nn.Module):
    """
    [DEPRECATED] See experiments/GraphClassifier for latest usage example.
    The layer for computing the wave equation solution.
    """
    def __init__(self, edge_index, edge_weight, degree=100, c=1., N=1000):
        """
        initiallized the ChebyLayer.

        Args:
            edge_index (tensor): shape (2, E). E is number of edges.
            edge_weight (tensor, optional): shape (E, ).
            degree (int): the degree of the Chebyshev polynomials. (the min power is 0, the max power is degree - 1)
            c (float): wave speed, optional, default=1.0
                the wave eqn is d^2u/dt^2 = - c^2 L u.
            N (int, optional): the number of points to be used in the integration. Defaults to 1000.
        """
        super(WaveLayer, self).__init__()
        self.cheby_layer = ChebyLayer(edge_index=edge_index, edge_weight=edge_weight)
        self.degree = degree
        self.N = N
        self.c = c

    def forward(self, x, y, ts):
        """
        compute the wave equation solution.

        Args:
            ts (tensor): shape (T, ). the sample time points.
            x (tensor): shape (n, *, *). the initial position.
            y  (tensor): shape (n, *, *). the initial speed.

        Returns:
            tensor: shape (T, n, *, *). the wave equation solution.
        """
        coefs_wave_x, coefs_wave_y = get_cheby_coefs_wave(ts, degree=self.degree, c=self.c, N=self.N)
        yWave_x = self.cheby_layer(x, coefs_wave_x)
        yWave_y = self.cheby_layer(y, coefs_wave_y)
        return yWave_x + yWave_y

def get_cheby_coefs_heat(ts, degree, device, c=1., N=1000):
    """
    get the Chebyshev polynomial coefficients of the heat equation.

    Args:
        ts (tensor): shape (T, ). the sample time points.
        degree (int): the degree of the Chebyshev polynomials. (the min power is 0, the max power is degree - 1)
        c (float, optional): the coefficient of the heat equation. Defaults to 1.
            the heat eqn is du/dt = - c^2 L u. we use c^2 to make sure the exponential does not explode.
        N (int, optional): the number of points to be used in the integration. Defaults to 1000.

    Returns:
        tensor: shape (T, k). the coefficients of the Chebyshev polynomials.
    """
    return get_cheby_coefs(heat_eqn_oper, c*c*ts, degree, device, N=N)

def get_cheby_coefs_wave(ts, degree, device, c=1., N=1000):
    """
    get the Chebyshev polynomial coefficients of the wave equation.

    Args:
        ts (tensor): shape (T, ). the sample time points.
        degree (int): the degree of the Chebyshev polynomials. (the min power is 0, the max power is degree - 1)
        c (float): wave speed, optional, default=1.0
            the wave eqn is d^2u/dt^2 = - c^2 L u.
        N (int, optional): the number of points to be used in the integration. Defaults to 1000.

    Returns:
        tuple: the coefficients of the Chebyshev polynomials.
            the first element is the coefficients of the wave equation for x. shape (T, k)
            the second element is the coefficients of the wave equation for y. shape (T, k)
    
    """
    return get_cheby_coefs(wave_eqn_oper_x, c*ts, degree, device, N=N), get_cheby_coefs(wave_eqn_oper_y, c*ts, degree, device, N=N)

def heat_eqn_oper(t, lam):
    """
    the function for computing the heat equation solution's operator e^{-tL}.

    Args:
        t (tensor): shape (T, ). the time points.
        lam (tensor): shape (N, ). the sampled points of eigenvalues of the laplacian matrix.

    Returns:
        tensor: shape (T, N). the heat equation solution result for each time t and each sample point of the eigenvalues.
    """
    return torch.exp(- t.view(-1, 1) * lam)

def wave_eqn_oper_x(t, lam):
    """
    the function for computing the wave equation solution's operator on x (initial position) cos(sqrt(lam) * t)

    Args:
        t (tensor): shape (T, ). the time points.
        lam (tensor): shape (N, ). the sampled points of eigenvalues of the laplacian matrix.

    Returns:
        tensor: shape (T, N). the wave equation solution result for each time t and each sample point of the eigenvalues.
    """
    lamsqrt = torch.sqrt(lam)
    tlamsqrt = t.view(-1, 1) * lamsqrt
    oper_x_res = torch.cos(tlamsqrt)
    return oper_x_res

def wave_eqn_oper_y(t, lam):
    """
    the function for computing the wave equation solution's operator on y (initial speed) sin(sqrt(lam) * t) / sqrt(lam)
    Note the t term is not needed for lam_0=0, because 1 equals to the limit of sin(a)/a at 0.

    Args:
        t (tensor): shape (T, ). the time points.
        lam (tensor): shape (N, ). the sampled points of eigenvalues of the laplacian matrix.

    Returns:
        tensor: shape (T, N). the wave equation solution result for each time t and each sample point of the eigenvalues.
    """
    lamsqrt = torch.sqrt(lam)
    tlamsqrt = t.view(-1, 1) * lamsqrt
    oper_y_res = torch.sin(tlamsqrt) / lamsqrt
    return oper_y_res