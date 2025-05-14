# src/graphormer_utils.py

import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import (
    ToUndirected,
    AddSelfLoops,
    NormalizeFeatures,
    Compose,
    BaseTransform,
)
from torch_geometric.utils import degree

class AddConstantEdgeAttr(BaseTransform):
    """Adds edge_attr = ones of dimension `dim` if missing (cheap, 𝑂(E))."""
    def __init__(self, dim: int = 1):
        self.dim = dim

    def __call__(self, data):
        if getattr(data, "edge_attr", None) is None:
            E = data.edge_index.size(1)
            data.edge_attr = torch.ones(E, self.dim)
        return data

class SafeAddShortestPathPE(BaseTransform):
    """
    Computes per-node one-hot shortest‑path encodings up to `max_dist`.
    Time: ~𝒪(N·(E+N)), Memory: 𝑂(N·(max_dist+1)).
    """
    def __init__(self, max_dist: int = 10, attr_name: str = "dist_pe"):
        self.max_dist = max_dist
        self.attr_name = attr_name

    def __call__(self, data):
        N = data.num_nodes
        # Build NetworkX graph for BFS
        G = nx.Graph()
        G.add_nodes_from(range(N))
        edges = list(zip(data.edge_index[0].tolist(),
                         data.edge_index[1].tolist()))
        G.add_edges_from(edges)

        # distance PE matrix (N, max_dist+1)
        dist_pe = torch.zeros((N, self.max_dist+1), dtype=torch.float)
        for i in range(N):
            lengths = nx.single_source_shortest_path_length(
                G, i, cutoff=self.max_dist)  # BFS cutoff  https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path_length.html
            for j, d in lengths.items():
                if d <= self.max_dist:
                    dist_pe[j, d] = 1.0
        setattr(data, self.attr_name, dist_pe)
        return data

def prepare_graphormer_dataset(
    dataset: InMemoryDataset,
    use_sp_pe: bool       = True,
    use_edge_attr: bool   = True,
    max_dist: int         = 10,
    edge_attr_dim: int    = 1
) -> InMemoryDataset:
    """
    One-line conversion to Graphormer format with toggles for large graphs:
      1) ToUndirected           (𝑂(E))                      https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.transforms.ToUndirected.html
      2) AddSelfLoops           (𝑂(N))                      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddSelfLoops.html
      3) SafeAddShortestPathPE (𝑂(N·(E+N))) if use_sp_pe  https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
      4) AddConstantEdgeAttr   (𝑂(E)) if use_edge_attr
      5) NormalizeFeatures     (𝑂(N·F))                    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.NormalizeFeatures.html
      6) Re-collate storage
    """
    # Build transform list dynamically
    tfms = [ToUndirected(), AddSelfLoops()]
    if use_sp_pe:
        tfms.append(SafeAddShortestPathPE(max_dist, "dist_pe"))
    if use_edge_attr:
        tfms.append(AddConstantEdgeAttr(dim=edge_attr_dim))
    tfms.append(NormalizeFeatures())

    composed = Compose(tfms)

    # Apply and rebuild
    data_list = [composed(data) for data in dataset]
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset