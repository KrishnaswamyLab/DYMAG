"""
[DEPRECATED]
"""
from torch_geometric.nn import MessagePassing

class PropogateSignal(MessagePassing):
    """
    just for computing mat-vec mult of the heat sol and the signal.
    """
    def __init__(self):
        """_summary_
        """
        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.

    def forward(self, x, edge_index, edge_weight):
        """_summary_

        Args:
            x (_type_): shape ()
            edge_index (_type_): _description_
            edge_weight (_type_, optional): _description_

        Returns:
            _type_: _description_
        """
        return self.propagate(edge_index=edge_index, edge_weight=edge_weight, size=None, x=x)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out