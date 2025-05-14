from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

def get_tudataset_with_kfold(name, fold_idx=0, num_folds=10, batch_size=32, seed=42):
    dataset = TUDataset(root=f'/tmp/{name}', name=name)
    labels = np.array([data.y.item() for data in dataset])
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = splits[fold_idx]
    train_dataset = dataset[train_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, dataset.num_features, dataset.num_classes


"""
Below is the code for the Laplacian + Random-Walk PE used by the GraphGPS method
-- you just need one additional step to convert any dataset: call this prepare_graphgps_dataset function
"""
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

from src.graphgps_data_utils import prepare_graphgps_dataset

def get_tudataset_with_kfold_pe(
    name,
    fold_idx=0,
    num_folds=10,
    batch_size=32,
    seed=42,
    use_pe=False,
):
    # 1) load raw TUDataset (or swap in your custom dataset class here)
    dataset = TUDataset(root=f"/tmp/{name}", name=name)

    # 2) convert/preprocess for GraphGPS
    dataset = prepare_graphgps_dataset(
        dataset, use_pe=use_pe, max_k=8, walk_len=16
    )

    # 3) stratified k‑fold split on dataset.data.y
    labels = np.array([data.y.item() for data in dataset])
    skf = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=seed
    )
    train_idx, test_idx = list(skf.split(np.zeros(len(labels)), labels))[fold_idx]

    train_dataset = dataset[train_idx.tolist()]
    test_dataset  = dataset[test_idx.tolist()]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.num_features, dataset.num_classes