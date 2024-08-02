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
        
        # Compute the moments for each k-hop sum for each node
        k_hop_sums = torch.zeros(x.size(0), x.size(1), self.K, self.M, x.size(2))
        for node_idx in range(x.size(1)):
            # get the k-hop subgraph for each node
            for k in range(1, self.K+1):
                subset, _, _, _ = k_hop_subgraph(node_idx, k, edge_index, relabel_nodes=True)
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
        
    def forward(self, x, edge_index):
        # x has shape [num_outputs, num_nodes, K, M, num_features]
        # want output to have shape [num_outputs, S, K, M, num_features]
        
        # Compute the moments for each node
        graph_moments = torch.zeros(x.size(0), self.S, x.size(2), x.size(3), x.size(4))
        for s in range(1, self.S+1):
            graph_moments[:, s-1, :, :, :] = self._get_graph_moment(x, s)
        return graph_moments
    
    def _get_graph_moment(self, x, s):
        # x has shape [num_outputs, num_nodes, K, M, num_features]
        # want output to have shape [num_outputs, K, M, num_features]
        return x.abs().pow(s).sum(dim=1)

if __name__ == '__main__':
    # test the aggregator
    x = torch.randn(5, 10, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)

    aggregator = KHopSumAggregator()
    output = aggregator(x, edge_index)
    print(output.size())
    graph_aggregator = GraphMomentAggregator()
    graph_output = graph_aggregator(output, edge_index)
    print(graph_output.size())
