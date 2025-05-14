import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj

from src.wavelet_conv import DiffusionConv, WaveletConv

def test_wavelet_conv_shapes():
    """
    Test that WaveletConv produces outputs with the expected shapes
    """
    # Create a small test graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                              [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
    num_nodes = 5
    feature_dim = 3
    
    # Create random node features
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    
    print("\nTesting default wavelet configuration (reshape=False):")
    conv = WaveletConv()
    out_default = conv(x, edge_index)
    
    # Expected shape: [num_nodes, feature_dim, num_wavelets]
    expected_shape = (num_nodes, feature_dim, 6)
    print(f"Expected shape: {expected_shape}, Actual shape: {out_default.shape}")
    assert out_default.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {out_default.shape}"
    
    print("\nTesting wavelet configuration with reshape=True:")
    conv_reshape = WaveletConv(reshape=True)
    out_reshape = conv_reshape(x, edge_index)
    
    # Expected shape: [num_nodes, num_wavelets * feature_dim]
    expected_shape = (num_nodes, 6 * feature_dim)
    print(f"Expected shape: {expected_shape}, Actual shape: {out_reshape.shape}")
    assert out_reshape.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {out_reshape.shape}"
    
    # Test custom scale list
    print("\nTesting custom scale list configuration:")
    scale_list = [0, 1, 2, 5, 10]
    conv_custom = WaveletConv(scale_list=scale_list)
    out_custom = conv_custom(x, edge_index)
    
    # Expected shape: [num_nodes, feature_dim, len(scale_list)]
    expected_shape = (num_nodes, feature_dim, len(scale_list))
    print(f"Expected shape: {expected_shape}, Actual shape: {out_custom.shape}")
    assert out_custom.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {out_custom.shape}"
    
    # Test custom aggregation
    print("\nTesting with custom aggregation (mean):")
    conv_mean = WaveletConv(aggr='mean')
    out_mean = conv_mean(x, edge_index)
    
    # Shape should be the same as default
    expected_shape = (num_nodes, feature_dim, 6)
    print(f"Expected shape: {expected_shape}, Actual shape: {out_mean.shape}")
    assert out_mean.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {out_mean.shape}"
    
    print("✅ Shape tests PASSED")
    return out_default, out_reshape, out_custom, out_mean

def test_wavelet_against_numpy():
    """
    Test WaveletConv results against numpy implementation
    """
    # Create a small test graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                              [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
    num_nodes = 5
    feature_dim = 3
    
    # Create random node features
    torch.manual_seed(42)
    x = torch.randn(num_nodes, feature_dim)
    x_np = x.numpy()
    
    # Convert to numpy for matrix-based computation
    A = to_scipy_sparse_matrix(edge_index).toarray()
    
    # Compute degree matrix
    degree = np.sum(A, axis=1)
    D_inv = np.diag(1.0 / degree)
    D_inv[np.isinf(D_inv)] = 0
    
    # Test with left normalization and default reshape=False
    print("\nTesting wavelet computation against numpy (left normalization, reshape=False):")
    
    # PyTorch Geometric implementation
    conv = WaveletConv(normalize="left", lazy=True)
    out_pyg = conv(x, edge_index).detach().numpy()
    
    # NumPy matrix implementation
    # Compute P = D^{-1}A
    P = D_inv @ A
    
    # Compute diffusion levels (I, P, P^2, ..., P^16)
    diffusion_levels = [x_np]
    for i in range(16):
        # For lazy diffusion: next_level = (I + P)/2 @ prev_level
        P_lazy = 0.5 * (np.eye(num_nodes) + P)
        next_level = P_lazy @ diffusion_levels[-1]
        diffusion_levels.append(next_level)
    
    # Stack diffusion levels
    diffusion_stack = np.stack(diffusion_levels, axis=0)  # Shape: [17, num_nodes, feature_dim]
    
    # Get wavelet constructor matrix
    wavelet_matrix = conv.wavelet_constructor.cpu().numpy()  # Shape: [6, 17]
    
    # Compute wavelet coefficients
    # Result dimensions will be [num_nodes, feature_dim, 6]
    wavelet_coeffs = np.einsum("ij,jkl->kli", wavelet_matrix, diffusion_stack)
    
    # Check if results match (no reshape needed)
    max_diff = np.abs(out_pyg - wavelet_coeffs).max()
    print(f"Maximum difference: {max_diff:.8f}")
    
    tolerance = 1e-5
    if max_diff < tolerance:
        print("✅ Test PASSED")
    else:
        print("❌ Test FAILED")
        print("PyG output shape:", out_pyg.shape)
        print("NumPy output shape:", wavelet_coeffs.shape)
        
        # Helpful debugging information for first two nodes
        print("\nFirst two nodes, first few values:")
        print("PyG:", out_pyg[:2, :2, :2])
        print("NumPy:", wavelet_coeffs[:2, :2, :2])
        
        # Check which wavelets have the largest differences
        diff = np.abs(out_pyg - wavelet_coeffs)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\nLargest difference at node {max_diff_idx[0]}, feature {max_diff_idx[1]}, wavelet {max_diff_idx[2]}")
        print(f"PyG value: {out_pyg[max_diff_idx]}, NumPy value: {wavelet_coeffs[max_diff_idx]}")
    
    # Test with reshape=True
    print("\nTesting wavelet computation against numpy (left normalization, reshape=True):")
    
    # PyTorch Geometric implementation
    conv_reshape = WaveletConv(normalize="left", lazy=True, reshape=True)
    out_pyg_reshape = conv_reshape(x, edge_index).detach().numpy()
    
    # Reshape NumPy result to match
    out_np_reshape = wavelet_coeffs.reshape(num_nodes, -1)
    
    # Check if results match
    max_diff = np.abs(out_pyg_reshape - out_np_reshape).max()
    print(f"Maximum difference: {max_diff:.8f}")
    
    if max_diff < tolerance:
        print("✅ Test PASSED")
    else:
        print("❌ Test FAILED")

if __name__ == "__main__":
    print("Testing WaveletConv implementation")
    print("=" * 60)
    # First test shapes
    test_wavelet_conv_shapes()
    # Then test numerical accuracy
    test_wavelet_against_numpy() 