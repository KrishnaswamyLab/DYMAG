"""
[DEPRECATED]
"""
import numpy as np
import scipy.sparse
import torch
from torch_geometric.nn import MessagePassing
from .propogate_signal import PropogateSignal
from .utils import safe_sqrt, safe_P2_eigh

class PDESolutionOperator():
    """
    First do eigendeconposition and then compute on the spectrum, so we can batch t.
    TODO which is faster, using scipy.sparse.expm sequentially or pytorch dense eigendecompositon over t?
    """
    def __init__(self, laplacian, device, eps=1e-5) -> None:
        """
        I will compute the eigendecomposition here.

        Args:
            laplacian (_type_): _description_
            device (_type_): _description_
        """
        self.laplacian = laplacian
        self.propogator = PropogateSignal()
        self.device = device
        if isinstance(laplacian, scipy.sparse.csc_matrix):
            laplacian = torch.tensor(laplacian.toarray(), dtype=torch.float).to(device)
        elif isinstance(laplacian, np.ndarray):
            laplacian = torch.tensor(laplacian, dtype=torch.float).to(device)
        else:
            laplacian = laplacian.float().to(device)
        ## needs the laplacian to be symmetric, otherwise use torch.linalg.eig
        self.lam, self.V = torch.linalg.eigh(laplacian)
        self.sqrtlam = safe_sqrt(self.lam, eps)
        self.eps = eps

    def getHeatEqnSolOperEigen(self, t):
        """
        see lemma 1.
        Returning the eigenvalues to save computation.
        Args:
            t (tensor): shape (m). m is the number of time points. 

        Returns:
            eigenvalues (tensor): shape (m, n). n is the number of nodes.
        """
        lamts = - t.unsqueeze(-1) * self.lam ## shape (m, n) 
        eigenvalues = torch.exp(lamts)
        # Ps = torch.einsum('ij,tj,kj->tik', self.V, torch.exp(lamts), self.V)
        return eigenvalues
    
    def getWaveEqnSolOperEigen(self, t):
        """
        see lemma 2.
        Returning the eigenvalues to save computation.

        Args:
            t (tensor): shape (m). m is the number of time points. 

        Returns:
            eigenvalues1, eigenvalues2 (tensor): shape (m, n). n is the number of nodes.
                eigenvalues of P1 and P2.
        """
        lamts1 = torch.cos(t.unsqueeze(-1) * self.sqrtlam)
        lamts2 = safe_P2_eigh(self.sqrtlam, t, self.eps)
        return lamts1, lamts2
