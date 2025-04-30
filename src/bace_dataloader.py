import torch
import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from torch_geometric.data import Data, DataLoader

def get_bace_dataloaders(batch_size=32):
    """
    Load and preprocess BACE classification dataset, returning PyTorch Geometric DataLoaders
    
    Args:
        batch_size (int): Batch size for the dataloaders. Defaults to 32.
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader) - PyTorch Geometric DataLoaders
    """
    # Load BACE dataset using DeepChem
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    tasks, dataset, transformers = dc.molnet.load_bace_classification(featurizer=featurizer)
    train, valid, test = dataset

    # Get train, validation and test data
    x_train, y_train, w_train, _ = train.X, train.y, train.w, train.ids
    x_valid, y_valid, w_valid, _ = valid.X, valid.y, valid.w, valid.ids
    x_test, y_test, w_test, _ = test.X, test.y, test.w, test.ids

    # Create PyG datasets
    train_data_list = []
    for i in range(len(x_train)):
        data = Data(
            x=torch.FloatTensor(x_train[i].node_features),
            edge_index=torch.LongTensor(x_train[i].edge_index),
            edge_attr=torch.FloatTensor(x_train[i].edge_features),
            y=torch.FloatTensor([y_train[i]]),
            w=torch.FloatTensor([w_train[i]])
        )
        train_data_list.append(data)

    valid_data_list = []
    for i in range(len(x_valid)):
        data = Data(
            x=torch.FloatTensor(x_valid[i].node_features),
            edge_index=torch.LongTensor(x_valid[i].edge_index),
            edge_attr=torch.FloatTensor(x_valid[i].edge_features),
            y=torch.FloatTensor([y_valid[i]]),
            w=torch.FloatTensor([w_valid[i]])
        )
        valid_data_list.append(data)

    test_data_list = []
    for i in range(len(x_test)):
        data = Data(
            x=torch.FloatTensor(x_test[i].node_features),
            edge_index=torch.LongTensor(x_test[i].edge_index),
            edge_attr=torch.FloatTensor(x_test[i].edge_features),
            y=torch.FloatTensor([y_test[i]]),
            w=torch.FloatTensor([w_test[i]])
        )
        test_data_list.append(data)

    # Create and return dataloaders
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader