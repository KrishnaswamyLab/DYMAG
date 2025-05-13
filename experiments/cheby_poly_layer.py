"""
Use Chebyshev polynomials for fast computation of the heat/wave equation solutions.
This file contains the implementation of the ChebyLayer class.

***
April 2023

for the chebyshev polynomials, refer to:
https://en.wikipedia.org/wiki/Chebyshev_polynomials

written as a pytorch_geometric layer.

"""
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian

class ChebyPolyLayer(MessagePassing):
    """
    Chebyshev polynomials as a custom message passing layer.
    Pass edge_index and edge_weight to the forward method.
    """
    def __init__(self, coefs=None):
        """
        the coefs is fixed for the layer (which is determined by the equation as well as the sampling points)
        can make more flexible by passing time points to forward method. computing the coefficients should'nt be slow.
        Args:
            coefs (tensor): shape (T, k). 
                k is the degree of the chebyshev polynomials.
                T is the number of time points. t is the sample times.
        """
        super().__init__(aggr="add", node_dim=-2)  # "Add" aggregation.
        self.coefs = coefs

    def forward(self, edge_index, edge_weight, x, coefs=None):
        """Evaluate the chebyshev polynomials through recursive message passing.
        coefs is 2 dimentional with the first dimension to be time points.

        Args:
            edge_index (tensor): shape (2, E). E is number of edges.
            edge_weight (tensor, optional): shape (E, ).
            x (tensor): shape (n, m). n is number of nodes. m is number of features.
            coefs (tensor): shape (T, k). defaults to None (use the coef specified at input).
                k is the degree of the chebyshev polynomials.
                T is the number of time points. t is the sample times.

        Returns:
            tensor: shape (T, n, m).
        """
        if coefs is None: coefs = self.coefs
        assert coefs is not None
        k = coefs.size(1)
        assert  k > 2
        ## using symmetrically normalized laplacian so that the eigenvalues are within [0, 2]
        ## see https://math.stackexchange.com/questions/2511544/largest-eigenvalue-of-a-normalized-graph-laplacian
        laplacian_edge_index, laplacian_edge_weight = get_laplacian(
            edge_index, edge_weight, normalization='sym')
        T0 = x
        out = coefs[:, 0].view(-1, 1, 1) * T0
        T1 = self.propagate(edge_index=laplacian_edge_index, x=x, edge_weight=laplacian_edge_weight)
        out += coefs[:, 1].view(-1, 1, 1) * T1

        for i in range(2, k):
            T2 = 2 * self.propagate(edge_index=laplacian_edge_index, x=T1, edge_weight=laplacian_edge_weight) - T0
            out += coefs[:, i].view(-1, 1, 1) * T2
            T0, T1 = T1, T2

        return out
        # return self.propagate(edge_index=laplacian_edge_index, x=x, edge_weight=laplacian_edge_weight)

    def message(self, x_j, edge_weight):
        """
        edge_weight is the edge weight of the graph laplacian.
        """
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out - x ## reparameterize lambda to range (-1, 1) (L_tilde = L - I, so L_tilde x = L x - x)

class ChebyLayer(MessagePassing):
    """
    [DEPRECATED.]
    Chebyshev polynomials as a custom message passing layer.
    """
    def __init__(self, edge_index, edge_weight):
        """computes the graph laplacian.

        Args:
            edge_index (tensor): shape (2, E). E is number of edges.
            edge_weight (tensor, optional): shape (E, ).
        """
        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        ## using symmetrically normalized laplacian so that the eigenvalues are within [0, 2]
        ## see https://math.stackexchange.com/questions/2511544/largest-eigenvalue-of-a-normalized-graph-laplacian
        self.laplacian_edge_index, self.laplacian_edge_weight = get_laplacian(
            self.edge_index, self.edge_weight, normalization='sym')

    def forward(self, x, coefs):
        """Evaluate the chebyshev polynomials through recursive message passing.
        coefs is 2 dimentional with the first dimension to be time points.

        Args:
            x (tensor): shape (n, *, *). n is number of nodes.
            coefs (tensor): shape (T, k). 
                k is the degree of the chebyshev polynomials.
                T is the number of time points. t is the sample times.

        Returns:
            tensor: shape (T, n, *, *).
        """
        k = coefs.size(1)
        assert  k > 2
        T0 = x
        out = coefs[:, 0].view(-1, 1, 1, 1) * T0
        T1 = self.propagate(edge_index=self.laplacian_edge_index, x=x, edge_weight=self.laplacian_edge_weight)
        out += coefs[:, 1].view(-1, 1, 1, 1) * T1

        for i in range(2, k):
            T2 = 2 * self.propagate(edge_index=self.laplacian_edge_index, x=T1, edge_weight=self.laplacian_edge_weight) - T0
            out += coefs[:, i].view(-1, 1, 1, 1) * T2
            T0, T1 = T1, T2

        return out
        # return self.propagate(edge_index=self.laplacian_edge_index, x=x, edge_weight=self.laplacian_edge_weight)

    def message(self, x_j):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return self.laplacian_edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out - x ## reparameterize lambda to range (-1, 1) (L_tilde = L - I, so L_tilde x = L x - x)
    
def get_cheby_coefs(func, ts, degree, device, N=1000):
    """
    get the Chebyshev polynomial coefficients of the function.

    Args:
        func (function): the function to be integrated.
            f(t, lam) has two parameters:
            t is the time point, and lam is the eigenvalue of the laplacian matrix
        ts (tensor): shape (T, ). the sample time points.
        degree (int): the degree of the Chebyshev polynomials. (the min power is 0, the max power is degree - 1)
        N (int, optional): the number of points to be used in the integration. Defaults to 1000.

    Returns:
        tensor: shape (T, k). the coefficients of the Chebyshev polynomials.
    
    Note:
        see https://en.wikipedia.org/wiki/Chebyshev_polynomials#Orthogonality for the formula for computing the coefficients.
    """
    ks = torch.arange(N).to(device)
    xks = torch.cos(torch.pi * (ks + 0.5) / (N)) + 1 ## reparameterize from [-1, 1] to [0, 2]
    ns = torch.arange(degree).unsqueeze(-1).to(device)
    Tn_xks = torch.cos(ns * torch.pi * (ks + 0.5) / (N)) ## shape (degree, N)
    func_xks = func(ts, xks).to(device) ## shape (T, N)
    coefs = (2 * Tn_xks.unsqueeze(0) * func_xks.unsqueeze(1)).mean(axis=-1) ## shape (T, degree)
    coefs[:, 0] /= 2
    return coefs

