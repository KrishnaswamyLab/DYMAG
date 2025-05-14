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