# the output of PDE_layer is a tensor of shape (num_outputs, num_nodes, num_features)
# make an aggregator that sums the values in the 1, 2, ..., K hop neighborhoods of each node
# the result should be a tensor of shape (num_outputs, num_nodes, K, num_features)

import torch
from torch_geometric.utils import k_hop_subgraph
#from torch_geometric.nn import GCNConv

class KHopSumAggregator(torch.nn.Module):
    def __init__(self, K=3, M=4): # M and K referenced from appendix B.2 of the paper
        super(KHopSumAggregator, self).__init__()  
        self.K = K # max number of hops
        self.M = M # largest moment for aggregation
        
    def forward(self, x, edge_index):
        # x has shape [num_outputs (t), num_nodes, num_features]
        # want output to have shape [num_outputs, num_nodes, K, M, num_features]
        num_nodes = edge_index.max().item() + 1
        # Compute the moments for each k-hop sum for each node
        k_hop_sums = torch.zeros(x.size(0), x.size(1), self.K, self.M, x.size(2))
        for node_idx in range(x.size(1)):
            if node_idx >= num_nodes: #, "Node index is out of bounds."
                    # this is an isolated node
                    # add in a self loop
                    self_loop = torch.tensor([[node_idx], [node_idx]], dtype=edge_index.dtype, device=edge_index.device)
                    edge_index = torch.cat([edge_index, self_loop], dim=1)
            # get the k-hop subgraph for each node
            for k in range(1, self.K+1):
                subset, _, _, _ = k_hop_subgraph(node_idx, k, edge_index, relabel_nodes=False)
                #print(f"Subset for node {node_idx}, k={k}: {subset}")
                for m in range(1, self.M+1):
                    #print(k,m)
                    k_hop_sums[:, node_idx, k-1, m-1,:] =  self._get_moment_sum_aggr(x[:,subset,:], m)
        return k_hop_sums

    def _get_moment_sum_aggr(self, x, m):
        # sum absolute value of x to the mth power
        # x has shape [num_outputs, |N(node)|, num_features]
        return x.abs().pow(m).sum(dim=1)
    
class GraphMomentAggregator(torch.nn.Module):
    def __init__(self, S=4):
        super(GraphMomentAggregator, self).__init__()
        self.S = S
        
    def forward(self, x, batch_index):
        # x has shape [num_outputs, num_nodes, K, M, num_features]
        # batch_index has shape [num_nodes]
        # We want output to have shape [num_graphs, num_outputs, S, K, M, num_features]
        
        # Get the number of graphs from the batch index
        num_graphs = batch_index.max().item() + 1
        
        # Initialize the output tensor
        num_outputs, num_nodes_total, K, M, num_features = x.size()
        graph_moments = torch.zeros(num_graphs, num_outputs, self.S, K, M, num_features, device=x.device)
        
        # Compute moments for each graph in the batch
        for g in range(num_graphs):
            # Get node indices for the current graph
            mask = (batch_index == g)
            
            # Extract relevant data for the current graph
            x_graph = x[:, mask]  # Shape [num_outputs, num_nodes_in_graph, K, M, num_features]
            
            for s in range(1, self.S+1):
                # Compute the s-th moment for the current graph
                graph_moments[g, :, s-1, :, :, :] = self._get_graph_moment(x_graph, s)
                
        return graph_moments
    
    def _get_graph_moment(self, x, s):
        # x has shape [num_outputs, num_nodes, K, M, num_features]
        # Output should have shape [num_outputs, K, M, num_features]
        return x.abs().pow(s).sum(dim=1)

if __name__ == '__main__':
    # Example usage
    num_outputs = 2
    num_nodes = 10
    K = 3
    M = 5
    num_features = 4
    batch_size = 5
    S = 4

    # Create example input
    x = torch.rand((num_outputs, num_nodes, K, M, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_nodes))
    batch_index = torch.randint(0, batch_size, (num_nodes,))

    # Initialize the aggregator
    aggregator = GraphMomentAggregator(S=S)

    # Compute the graph moments
    graph_moments = aggregator(x, batch_index)

    print("Graph moments shape:", graph_moments.shape)
