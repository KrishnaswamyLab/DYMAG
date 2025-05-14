import torch
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from torch_geometric.transforms import (
    ToUndirected,
    AddSelfLoops,
    NormalizeFeatures,
    Compose,
    AddRandomWalkPE,
)
from torch_geometric.utils import get_laplacian
from torch_geometric.data import InMemoryDataset

class SafeAddLaplacianEigenvectorPE:
    """
    Computes up to `max_k` Laplacian eigenvectors per graph,
    pads to fixed width so you always get (N, max_k).
    """
    def __init__(self, max_k: int = 8, attr_name: str = "lap_pe"):
        self.max_k = max_k
        self.attr_name = attr_name

    def __call__(self, data):
        N = data.num_nodes
        k = min(self.max_k, max(0, N - 1))

        if k == 0:
            pe = torch.zeros((N, self.max_k), dtype=torch.float)
        else:
            edge_index, edge_weight = get_laplacian(
                data.edge_index, normalization="sym", num_nodes=N
            )
            L = sp.coo_matrix(
                (edge_weight.numpy(), (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(N, N),
            )
            try:
                vals, vecs = splinalg.eigsh(L, k=k + 1, which="SM")
                vecs = vecs[:, 1 : k + 1]  # drop trivial eigenvector
                pe = torch.from_numpy(vecs).float()
            except Exception:
                pe = torch.zeros((N, k), dtype=torch.float)

            if k < self.max_k:
                pad = torch.zeros((N, self.max_k - k), dtype=torch.float)
                pe = torch.cat([pe, pad], dim=1)

        setattr(data, self.attr_name, pe)
        return data


def prepare_graphgps_dataset(
    dataset: InMemoryDataset,
    use_pe: bool = False,
    max_k: int = 8,
    walk_len: int = 16,
) -> InMemoryDataset:
    """
    Given any InMemoryDataset (e.g. TUDataset or your custom dataset),
    applies GraphGPS-style preprocessing *in place*:

      1) ToUndirected
      2) AddSelfLoops
      3) (optional) Laplacian PE (k=max_k) + RandomWalk PE (len=walk_len)
      4) NormalizeFeatures

    After calling this, each Data in dataset will have:
      - data.edge_index updated
      - if use_pe: data.lap_pe (N, max_k), data.rw_pe (N, walk_len)
    """
    base = [ToUndirected(), AddSelfLoops()]
    pe_tfms = []
    if use_pe:
        pe_tfms = [
            SafeAddLaplacianEigenvectorPE(max_k=max_k, attr_name="lap_pe"),
            AddRandomWalkPE(walk_length=walk_len, attr_name="rw_pe"),
        ]
    tail = [NormalizeFeatures()]

    transform = Compose(base + pe_tfms + tail)

    # apply to every graph
    data_list = []
    for data in dataset:
        data = transform(data)
        data_list.append(data)

    # overwrite the dataset internals
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset