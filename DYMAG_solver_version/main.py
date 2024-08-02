from .data_module import RandomDataset
from torch_geometric.data import Data
import numpy as np
import torch
import networkx as nx
from .PDE_layer import DYMAG, heat_derivative_func

# run DYMAG on a RandomDataset of Erdos-Renyi graphs

# TODO: set up training loop on this dataset
