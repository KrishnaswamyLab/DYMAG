"""
[DEPRECATED]
DEPRECATED. Use sol_propagator.PDESolutionPropogator instead.
"""
import numpy as np
from scipy.sparse.linalg import expm
import scipy.sparse
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import from_scipy_sparse_matrix
from abc import ABC, abstractmethod

device = torch.device("cuda")

class EqnSol(nn.Module, ABC):
    """
    one part computes the solution.
    the other part computes the propagation using pyg. no trainable parameters
    TODO make available broadcasting across time for efficiency.
    """
    def __init__(self, laplacian, device) -> None:
        """_summary_

        Args:
            laplacian (_type_): scipy.sparse matrix (csc)
        TODO possible speed-ups: 
            1. cache exp(-Lt) for ts.
            2. compute for different ts in parallel?
            3. which is faster, using scipy.sparse or pytorch?
        """
        super().__init__()
        self.laplacian = laplacian
        self.propogator = PropogateSignal()
        self.device = device

class HeatEqnSolSparseExpm(EqnSol):
    """
    computes the heat equation solution. exp(-tL)
    """
    def forward(self, x, t):
        """
        see lemma 1.

        Args:
            x (tensor): should be shape (n, *, *), n is number of nodes.
            t (float): time.

        Returns:
            _type_: _description_
        """
        if isinstance(self.laplacian, torch.Tensor):
            self.laplacian = scipy.sparse.csc_matrix(self.laplacian.cpu().numpy())           
        elif isinstance(self.laplacian, np.ndarray):
            self.laplacian = scipy.sparse.csc_matrix(self.laplacian)
        Pt = expm(- t * self.laplacian)
        edge_index, edge_weight = from_scipy_sparse_matrix(Pt)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        return self.propogator(x, edge_index, edge_weight)

class EqnSolEigdecomp(EqnSol):
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
        super().__init__(laplacian, device)
        if isinstance(laplacian, scipy.sparse.csc_matrix):
            laplacian = torch.tensor(laplacian.toarray(), dtype=torch.float).to(device)
        elif isinstance(laplacian, np.ndarray):
            laplacian = torch.tensor(laplacian, dtype=torch.float).to(device)
        else:
            laplacian = laplacian.float().to(device)
        ## needs the laplacian to be symmetric, otherwise use torch.linalg.eig
        self.lam, self.V = torch.linalg.eigh(laplacian)
        self.eps = eps


class HeatEqnSol(EqnSolEigdecomp):
    def getP(self, t):
        """
        see lemma 1.

        Args:
            t (tensor): shape (m) m is the number of time points. 

        Returns:
            _type_: _description_
        """
        lamts = t.unsqueeze(-1) * self.lam ## shape (m, n) n is the number of nodes.
        Ps = torch.einsum('ij,tj,kj->tik', self.V, torch.exp(lamts), self.V)

    def forward(self, x, t):
        """
        see lemma 1.

        Args:
            x (_type_): _description_
            t (tensor): shape (m) m is the number of time points. 

        Returns:
            _type_: _description_
        """
        Ps = self.getP(t)
        NotImplemented


class WaveEqnSol(EqnSolEigdecomp):
    """
    one part computes the laplacian's sqrt and sin and cosing finctions.
    the other part computes the propagation using pyg. no trainable parameters
    TODO it seems to involve eigendecomposition, which is slow.
    """
    def __init__(self, laplacian, device, eps=1e-5) -> None:
        """
        I will compute the eigendecomposition here.

        Args:
            laplacian (_type_): _description_
            device (_type_): _description_
        """
        super().__init__(laplacian, device, eps)
        self.sqrtlam = safe_sqrt(self.lam, self.eps)

    def forward(self, x, y, t):
        """
        see lemma 2.

        Args:
            x (_type_): _description_
            y (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        P1 = self.V @ torch.diag(torch.cos(t * self.sqrtlam)) @ self.V.T
        assert torch.abs(self.sqrtlam[0]) < self.eps, "sqrtlam[0] is not zero"
        ## the t L term is included as when i=1, the lam_1=0, and the sin(t*lam_1)/sqrt(lam_1) is t.
        P2 = self.V @ torch.diag(safe_P2_eigh(self.sqrtlam, t, self.eps)) @ self.V.T
        edge_index, edge_weight = from_tensor(P1)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        res1 = self.propogator(x, edge_index, edge_weight)
        edge_index, edge_weight = from_tensor(P2)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        res2 = self.propogator(y, edge_index, edge_weight)
        return res1 + res2

def safe_sqrt(lam, eps):
    """
    if -eps < lam < 0 return 0.
    if lam < -eps raise exception. 
    else return sqrt(lam).
    """
    assert (lam > - eps).all(), "lam is not positive"
    res = torch.sqrt(lam)
    res[lam < 0.] = 0.
    return res

def safe_P2_eigh(sqrtlam, t, eps):
    """
    Computes sin(t * sqrtlam) / sqrtlam.
    if |sqrtlam|<eps, return t, the limit as sqrtlam->0.
    sqrtlam is a torch tensor, and t is a number (or a torch tensor).
    """
    res = torch.zeros_like(sqrtlam)
    smalllam_ids = torch.abs(sqrtlam) < eps
    res[smalllam_ids] = t
    res[~smalllam_ids] = torch.sin(t * sqrtlam[~smalllam_ids]) / sqrtlam[~smalllam_ids]
    return res

def from_tensor(A):
    """
    Convert a torch tensor to edge_index and edge_weight. using pytorch.
    """
    rows, cols = torch.nonzero(A, as_tuple=True)
    edge_index = torch.stack((rows, cols), dim=0)
    edge_weight = A[rows, cols]
    return edge_index, edge_weight

class PropogateSignal(MessagePassing):
    """
    just for computing mat-vec mult of the heat sol and the signal.
    """
    def __init__(self):
        """_summary_
        """
        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.

    def forward(self, x, edge_index, edge_weight):
        """_summary_

        Args:
            x (_type_): shape ()
            edge_index (_type_): _description_
            edge_weight (_type_, optional): _description_

        Returns:
            _type_: _description_
        """
        return self.propagate(edge_index=edge_index, edge_weight=edge_weight, size=None, x=x)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out