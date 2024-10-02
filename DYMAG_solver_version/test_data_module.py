import unittest
from data_module import RandomDataset
import networkx as nx
import torch_geometric

class TestRandomDataset(unittest.TestCase):
    def test_generate_graphs(self):
        # Create a RandomDataset instance
        dataset = RandomDataset(random_graph_model='er', num_graphs=5, graph_params_bounds={'n': (10, 20), 'p': (0.1, 0.5)})

        # Check if the number of generated graphs matches the specified number of graphs
        self.assertEqual(len(dataset.graphs), 5)

        # Check if the generating parameters are correctly stored
        self.assertEqual(len(dataset.generating_parameters), 5)

        # Check if each generated graph is of type networkx.Graph
        for graph in dataset.graphs:
            self.assertIsInstance(graph, nx.Graph)

    def test_sample_graph_params(self):
        # Create a RandomDataset instance
        dataset = RandomDataset(random_graph_model='er', num_graphs=1, graph_params_bounds={'n': (10, 20), 'p': (0.1, 0.5)})

        # Sample graph parameters
        graph_params = dataset.sample_graph_params()

        # Check if the sampled graph parameters are within the specified bounds
        self.assertGreaterEqual(graph_params['n'], 10)
        self.assertLessEqual(graph_params['n'], 20)
        self.assertGreaterEqual(graph_params['p'], 0.1)
        self.assertLessEqual(graph_params['p'], 0.5)

    def test_getitem(self):
        # Create a RandomDataset instance
        dataset = RandomDataset(random_graph_model='er', num_graphs=1, graph_params_bounds={'n': (10, 20), 'p': (0.1, 0.5)})

        # Get the graph data object at index 0
        data = dataset[0]

        # Check if the returned object is of type torch_geometric.data.Data
        self.assertIsInstance(data, torch_geometric.data.Data)

        # Check if the node features, target values, and edge indices are correctly set
        self.assertEqual(data.x.shape, (data.num_nodes, 1))
        self.assertEqual(data.y.shape, (data.num_nodes - 1, 1))
        self.assertEqual(data.edge_index.shape[0], 2)

if __name__ == '__main__':
    unittest.main()