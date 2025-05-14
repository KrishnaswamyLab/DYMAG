import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj

from src.wavelet_conv import DiffusionConv

def test_diffusion_conv():
    """
    Test DiffusionConv implementation against direct matrix computations
    for all normalization types (left, right, symmetric).
    """
    # Create a small test graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                              [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
    num_nodes = 5
    
    # Create random node features
    torch.manual_seed(42)
    x = torch.randn(num_nodes, 3)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index)
    
    # Convert to numpy for matrix-based computation
    x_np = x.numpy()
    A = to_scipy_sparse_matrix(edge_index).toarray()
    
    # Compute degree matrix
    degree = np.sum(A, axis=1)
    D_inv = np.diag(1.0 / degree)
    D_inv[np.isinf(D_inv)] = 0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    
    # Test each normalization type
    for normalize in ["left", "right", "symmetric"]:
        print(f"\nTesting {normalize} normalization:")
        
        # PyTorch Geometric implementation
        conv = DiffusionConv(normalize=normalize, lazy=True)
        out_pyg = conv(x, edge_index).detach().numpy()
        
        # NumPy matrix implementation
        if normalize == "left":
            # Left normalization: (I + D^{-1}A)/2
            P = D_inv @ A
            out_np = 0.5 * (x_np + P @ x_np)
        elif normalize == "right":
            # Right normalization: (I + AD^{-1})/2
            P = A @ D_inv
            out_np = 0.5 * (x_np + P @ x_np)
        elif normalize == "symmetric":
            # Symmetric normalization: (I + D^{-1/2}AD^{-1/2})/2
            P = D_inv_sqrt @ A @ D_inv_sqrt
            out_np = 0.5 * (x_np + P @ x_np)
        
        # Check if results match
        max_diff = np.abs(out_pyg - out_np).max()
        print(f"Maximum difference: {max_diff:.8f}")
        
        if max_diff < 1e-6:
            print("✅ Test PASSED")
        else:
            print("❌ Test FAILED")
            print("PyG output:")
            print(out_pyg)
            print("NumPy output:")
            print(out_np)

def test_nonlazy_diffusion():
    """
    Test non-lazy diffusion (lazy=False)
    """
    # Create a small test graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                              [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
    num_nodes = 5
    
    # Create random node features
    torch.manual_seed(42)
    x = torch.randn(num_nodes, 3)
    
    # Convert to numpy for matrix-based computation
    x_np = x.numpy()
    A = to_scipy_sparse_matrix(edge_index).toarray()
    
    # Compute degree matrix
    degree = np.sum(A, axis=1)
    D_inv = np.diag(1.0 / degree)
    D_inv[np.isinf(D_inv)] = 0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    
    print("\nTesting non-lazy diffusion (symmetric normalization):")
    
    # PyTorch Geometric implementation
    conv = DiffusionConv(normalize="symmetric", lazy=False)
    out_pyg = conv(x, edge_index).detach().numpy()
    
    # NumPy matrix implementation (no lazy - just D^{-1/2}AD^{-1/2})
    P = D_inv_sqrt @ A @ D_inv_sqrt
    out_np = P @ x_np
    
    # Check if results match
    max_diff = np.abs(out_pyg - out_np).max()
    print(f"Maximum difference: {max_diff:.8f}")
    
    if max_diff < 1e-6:
        print("✅ Test PASSED")
    else:
        print("❌ Test FAILED")

if __name__ == "__main__":
    print("Testing DiffusionConv against NumPy matrix implementations")
    print("=" * 60)
    test_diffusion_conv()
    test_nonlazy_diffusion() 