import torch

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
    res = torch.zeros(t.size(0), sqrtlam.size(0)).to(sqrtlam.device)
    smalllam_ids = torch.abs(sqrtlam) < eps
    res[:, smalllam_ids] = t.unsqueeze(-1)
    res[:, ~smalllam_ids] = torch.sin(t.unsqueeze(-1) * sqrtlam[~smalllam_ids]) / sqrtlam[~smalllam_ids]
    return res

def from_tensor(A):
    """
    Convert a torch tensor to edge_index and edge_weight. using pytorch.
    """
    rows, cols = torch.nonzero(A, as_tuple=True)
    edge_index = torch.stack((rows, cols), dim=0)
    edge_weight = A[rows, cols]
    return edge_index, edge_weight