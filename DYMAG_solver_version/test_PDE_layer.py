import torch
import networkx as nx
from PDE_layer import PDE_layer, heat_derivative_func

def test_PDE_layer():
    # Create a sample graph
    G = nx.cycle_graph(5)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.randn(5, 3)
    batch = torch.tensor([0, 0, 0, 0, 0])

    # Create a PDE_layer instance
    pde_layer = PDE_layer(derivative_func=heat_derivative_func)

    # Perform forward pass
    outputs = pde_layer(x, edge_index, batch)

    # Assert the shape of the outputs
    assert outputs.shape == (26, 5, 3)

    # Assert the values of the outputs
    assert torch.allclose(outputs[0], x)
    #assert torch.allclose(outputs[-1], x)
    print("PDE_layer test passed!")

def test_PDE_layer_batch():
    # Create multiple sample graphs
    G1 = nx.cycle_graph(5)
    G2 = nx.complete_graph(3)
    G3 = nx.star_graph(4)

    # Convert graphs to edge index format
    edge_index1 = torch.tensor(list(G1.edges)).t().contiguous()
    edge_index2 = torch.tensor(list(G2.edges)).t().contiguous()
    edge_index3 = torch.tensor(list(G3.edges)).t().contiguous()

    # Create node features for each graph
    x1 = torch.randn(5, 3)
    x2 = torch.randn(3, 3)
    x3 = torch.randn(4, 3)

    # Create batch tensor
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

    # Create a PDE_layer instance
    pde_layer = PDE_layer(derivative_func=heat_derivative_func)

    # Batch the graphs together
    batched_edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
    batched_x = torch.cat([x1, x2, x3], dim=0)

    # Perform forward pass
    outputs = pde_layer(batched_x, batched_edge_index, batch)
    # Assert the shape of the outputs
    # output takes on shape (num_steps, num_nodes, num_features)
    assert outputs.shape == (26, 12, 3)

    # Assert the values of the outputs
    #assert torch.allclose(outputs[0], x1)
    #assert torch.allclose(outputs[5], x2)
    #assert torch.allclose(outputs[15], x3)
    print("PDE_layer batch test passed!")


test_PDE_layer()
test_PDE_layer_batch()