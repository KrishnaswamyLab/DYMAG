import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np

class RandomDataset(Dataset):
    def __init__(self, random_graph_model, num_graphs, graph_params_bounds, node_features=None):
        super(RandomDataset, self).__init__()

        self.random_graph_model = random_graph_model
        self.num_graphs = num_graphs
        self.graph_params_bounds = graph_params_bounds

        self.graphs, self.generating_parameters = self.generate_graphs()
        self.node_features = node_features # options are degree or none

    def generate_graphs(self):
        graphs = []
        generating_parameters = []
        for _ in range(self.num_graphs):
            graph_params = self.sample_graph_params()
            if self.random_graph_model == 'er':
                graph = nx.erdos_renyi_graph(n=int(graph_params['n']), p=graph_params['p'])
            elif self.random_graph_model == 'sbm':
                sizes = [int(graph_params['n'] / graph_params['k'])] * int(graph_params['k'])
                p_matrix = np.full((len(sizes), len(sizes)), graph_params['p_out'])
                np.fill_diagonal(p_matrix, graph_params['p_in'])
                graph = nx.stochastic_block_model(sizes, p_matrix)
            else:
                raise ValueError("Invalid random graph model")
            graphs.append(graph)
            generating_parameters.append(graph_params)
        return graphs, generating_parameters

    def sample_graph_params(self):
        graph_params = {}
        for param, bounds in self.graph_params_bounds.items():
            lower_bound, upper_bound = bounds
            if param == 'n' or param == 'k':
                value = np.random.randint(lower_bound, upper_bound + 1)
            else:
                value = np.random.uniform(lower_bound, upper_bound)
            graph_params[param] = value
        return graph_params

    def len(self):
        return self.num_graphs

    def get(self, idx):
        graph = self.graphs[idx]
        params = self.generating_parameters[idx]
        if self.node_features == 'degree':
            x = np.array([graph.degree[node] for node in graph.nodes()]).reshape(-1, 1)
        else:
            x = np.eye(graph.number_of_nodes())
        y = np.array([params[param] for param in self.graph_params_bounds.keys() if param in params]).reshape(-1, 1)
        # store the graph structure so a GCN can use it
        data = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float))
        data.edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        return data
        #return Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float))

if __name__ == '__main__':
    random_graph_model = 'er'  # 'er' or 'sbm'
    num_graphs = 10
    graph_params_bounds = {
        'n': (50, 50),
        'p': (0.1, 0.5),
        'k': (1, 5),
        'p_in': (0.5, 0.9),
        'p_out': (0.05, 0.2)
    }

    dataset = RandomDataset(random_graph_model, num_graphs, graph_params_bounds)
    print(dataset[0])
