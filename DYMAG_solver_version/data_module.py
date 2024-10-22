import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
from torch_geometric.utils import to_undirected

class RandomDataset(Dataset):
    """
    A dataset class for generating random graphs.

    Args:
        random_graph_model (str): The type of random graph model to use. Options are 'er' (Erdos-Renyi) or 'sbm' (Stochastic Block Model).
        num_graphs (int): The number of graphs to generate.
        graph_params_bounds (dict): A dictionary specifying the bounds for each graph parameter. The keys are the parameter names and the values are tuples (lower_bound, upper_bound).
        node_features (str or None): The type of node features to include. Options are 'degree' or None.

    Attributes:
        random_graph_model (str): The type of random graph model.
        num_graphs (int): The number of graphs to generate.
        graph_params_bounds (dict): The bounds for each graph parameter.
        node_features (str or None): The type of node features.
        graphs (list): The generated graphs.
        generating_parameters (list): The generating parameters for each graph.
    """

    def __init__(self, random_graph_model, num_graphs, graph_params_bounds, node_features=None):
        super(RandomDataset, self).__init__()

        self.random_graph_model = random_graph_model
        self.num_graphs = num_graphs
        self.graph_params_bounds = graph_params_bounds
        self.node_features = node_features

        self.graphs, self.generating_parameters = self.generate_graphs()

    def generate_graphs(self):
        """
        Generates random graphs based on the specified parameters.

        Returns:
            graphs (list): The generated graphs.
            generating_parameters (list): The generating parameters for each graph.
        """
        graphs = []
        generating_parameters = []
        for _ in range(self.num_graphs):
            graph_params = self.sample_graph_params()
            if self.random_graph_model == 'er':
                graph = nx.erdos_renyi_graph(n=int(graph_params['n']), p=graph_params['p'])
            elif self.random_graph_model == 'sbm':
                sizes = [int(graph_params['n'] / graph_params['k'])] * int(graph_params['k'])
                # modify the size of the last block to make sure the total number of nodes is n
                sizes[-1] += graph_params['n'] - sum(sizes)
                p_matrix = np.full((len(sizes), len(sizes)), graph_params['p_out'])
                np.fill_diagonal(p_matrix, graph_params['p_in'])
                graph = nx.stochastic_block_model(sizes, p_matrix)
            else:
                raise ValueError("Invalid random graph model")
            graphs.append(graph)
            generating_parameters.append(graph_params)
        return graphs, generating_parameters

    def sample_graph_params(self):
        """
        Samples random graph parameters based on the specified bounds.

        Returns:
            graph_params (dict): The sampled graph parameters.
        """
        graph_params = {}
        for param, bounds in self.graph_params_bounds.items():
            lower_bound, upper_bound = bounds
            if param in ['n', 'k']:
                value = np.random.randint(lower_bound, upper_bound + 1)
            else:
                value = np.random.uniform(lower_bound, upper_bound)
            graph_params[param] = value
        return graph_params

    def __len__(self):
        """
        Returns the number of graphs in the dataset.

        Returns:
            int: The number of graphs.
        """
        return self.num_graphs

    def __getitem__(self, idx):
        """
        Returns the graph and its corresponding parameters at the given index.

        Args:
            idx (int): The index of the graph.

        Returns:
            Data: The graph data object containing the node features, target values, and edge indices.
        """
        graph = self.graphs[idx]
        params = self.generating_parameters[idx]
        if self.node_features == 'degree':
            x = np.array([graph.degree[node] for node in graph.nodes()]).reshape(-1, 1)
        if self.node_features == 'random':
            n_random_signal = self.graph_params_bounds.get('n_random_signal', 5)
            x = np.random.rand(graph.number_of_nodes(), n_random_signal)
        else:
            x = np.eye(graph.number_of_nodes())
        y = np.array([params[param] for param in self.graph_params_bounds.keys() if param in params and param != 'n']).reshape(-1, 1)
        
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        edge_index = to_undirected(edge_index) 

        return Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=edge_index)

if __name__ == '__main__':
    random_graph_model = 'sbm'  # 'er' or 'sbm'
    num_graphs = 10
    graph_params_bounds = {
        'n': (50, 50),
        'p': (0.1, 0.5),
        'k': (1, 5),
        'p_in': (0.5, 0.9),
        'p_out': (0.05, 0.2)
    }

    graph_params_bounds = {
        'n': (50, 50),
        'k': (1, 5),
        'p_in': (0.5, 0.9),
        'p_out': (0.05, 0.2)
    }

    dataset = RandomDataset(random_graph_model, num_graphs, graph_params_bounds)
    dataloader = DataLoader(dataset, batch_size = 5, shuffle=True)
    print(dataset[0])
    print(dataset[0].y)
    assert dataset[0].is_undirected()

    for batch in dataloader:
        print(batch)
