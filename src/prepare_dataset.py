"""
Prepare datasets 
- add edge weights to the original dataset.
- get folds for cross validation.

***
May 2023

"""

from sklearn.model_selection import KFold
import pickle
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from pathlib import Path

def prep_dataset_tud(n_split, dataset_name, data_root_path, save_path, rand_state):
    # Load the dataset
    original_dataset = TUDataset(root=data_root_path, name=dataset_name, use_node_attr=True)

    # Function to add equal edge weights to a graph
    def add_edge_weights(data):
        edge_index = data.edge_index
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float)
        return Data(edge_index=edge_index, x=data.x, y=data.y, edge_weight=edge_weight)

    # Create a new dataset with added edge weights
    dataset = [add_edge_weights(data) for data in original_dataset]

    kf = KFold(n_splits=n_split, shuffle=True, random_state=rand_state)
    id_folds = list(kf.split(dataset))

    # Create the directory
    ablation_dir = Path(save_path)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Save the dataset in the new directory
    with (ablation_dir / "dataset.pkl").open("wb") as f:
        pickle.dump(dataset, f)

    # Save the id_folds in the new directory
    with (ablation_dir / "id_folds.pkl").open("wb") as f:
        pickle.dump(id_folds, f)

if __name__ == "__main__":
    # Set the random state
    rand_state = 42

    # Set the number of splits
    n_split = 10
    root_path = "/gpfs/gibbs/pi/***/***/Graph_expressivity"
    
    data_root_path = f"{root_path}/data/tmp"

    # dataset_name = "ENZYMES"
    # save_path = f"{root_path}/data/{dataset_name}"  ## processed data. don't confuse with raw data.
    # prep_dataset_tud(n_split, dataset_name, data_root_path, save_path, rand_state)

    # dataset_name = "PROTEINS"
    # save_path = f"{root_path}/data/{dataset_name}"
    # prep_dataset_tud(n_split, dataset_name, data_root_path, save_path, rand_state)

    # dataset_name = "MUTAG"
    # save_path = f"{root_path}/data/{dataset_name}"
    # prep_dataset_tud(n_split, dataset_name, data_root_path, save_path, rand_state)

    dataset_name = "NCI1"
    save_path = f"{root_path}/data/{dataset_name}"
    prep_dataset_tud(n_split, dataset_name, data_root_path, save_path, rand_state)

