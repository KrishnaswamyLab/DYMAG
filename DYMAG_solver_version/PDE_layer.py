import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Batch
import torch.nn as nn

class PDE_layer(MessagePassing):
    """
    PDE_layer class represents a layer for solving partial differential equations (PDEs) using message passing.

    Args:
        derivative_func (callable): A function that computes the derivative of the PDE.

    Attributes:
        step_size (float): The step size for numerical integration.
        solver (str): The solver method for solving the PDE. Can be 'euler' or 'rk4'.
        sampling_interval (float): The interval at which to sample the solution.
        final_t (float): The final time for the integration.
        dynamics (str): A description of the time derivative of the PDE.

    Methods:
        get_laplacian: Computes the Laplacian of the input data.
        forward: Performs the forward pass of the PDE solver.

    """

    def __init__(self, dynamics='sprott', n_largest_graph=100, b=0.25, **kwargs):
        super(PDE_layer, self).__init__(aggr='add')
        self.step_size = .01
        self.solver = kwargs.get('solver', 'rk4')
        # set up sampling_interval and final_t from kwargs if provided
        self.sampling_interval = kwargs.get('sampling_interval', .2)
        self.final_t = kwargs.get('final_t', 5)
        self.b = b
        #self.random_weights = torch.rand((n_largest_graph, n_largest_graph)) - 0.5
        # set random weights to be +1 or -1
        self.random_weights = (torch.randint(0, 2, (n_largest_graph, n_largest_graph)) * 2 - 1) / (n_largest_graph -1)**(1/2)
        if dynamics == 'heat':
            self.derivative_func = self.heat_dynamic
        elif dynamics == 'sprott':
            self.derivative_func = self.sprott_dynamic
        print(f'Initialized with {dynamics} dynamics')

        self.output_times = torch.arange(0, self.final_t + self.sampling_interval, self.sampling_interval)

    def heat_dynamic(self, x, edge_index, batch):
        return -self.get_laplacian(x, edge_index, batch)
    
    def sprott_dynamic(self, x, edge_index, batch):
        """
        Sprott dynamics:
        du/dt = -b * u + tanh(sum_ij a_ij * u_j)
        """
        # row, col represent source and target nodes of directed edges
        row, col = edge_index

        # Create a map from batched node indices to original indices
        batch_node_count = (batch == 0).sum().item()  # assuming uniform size across graphs in the batch
        batch_offset = batch * batch_node_count

        # Adjust row and col indices by subtracting the batch offset
        row_adjusted = row - batch_offset[row]
        col_adjusted = col - batch_offset[col]

        # Propagate messages using random weights
        #weighted_x = self.random_weights[row_adjusted, col_adjusted][:, None] * x[col]

        # Use self.propagate to aggregate the messages
        aggregated_message = self.propagate(edge_index, x=x, norm = self.random_weights[row_adjusted, col_adjusted])

        # Apply Sprott dynamics
        dt = -self.b * x + torch.tanh(aggregated_message)
        return dt


    def get_laplacian(self, x, edge_index, batch, normalized=True):
        """
        Computes the Laplacian of the input data.

        Args:
            x (Tensor): The input data.
            edge_index (LongTensor): The edge indices of the graph.
            batch (LongTensor): The batch indices of the graph.
            normalized (bool): Whether to normalize the Laplacian.

        Returns:
            Tensor: The Laplacian of the input data.

        """
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        if normalized:
            deg = degree(row, num_nodes=x.size(0), dtype=torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = None

        adj = self.propagate(edge_index, x=x, norm=norm, size=(x.size(0), x.size(0)))
        return x - adj

    def forward(self, x, edge_index, batch):
        """
        Performs the forward pass of the PDE solver.

        Args:
            x (Tensor): The input data.
            edge_index (LongTensor): The edge indices of the graph.
            batch (LongTensor): The batch indices of the graph.

        Returns:
            Tensor: The solution of the PDE at different time steps. Output has shape [time_steps, num_nodes, num_features]
        """
        num_nodes = x.size(0)
        batch_size = batch.max().item() + 1

        if self.solver == 'euler':
            num_steps = int(self.final_t / self.step_size)
            sampling_interval_steps = int(self.sampling_interval // self.step_size)
            num_outputs = (num_steps // sampling_interval_steps) + 1

            outputs = torch.zeros((int(num_outputs), num_nodes, x.size(1)), device=x.device, requires_grad=False)
            outputs[0] = x

            output_idx = 1
            for t_step in range(1, num_steps + 1):
                dt = self.derivative_func(x, edge_index, batch)
                x = x + self.step_size * dt
                if t_step % sampling_interval_steps == 0:
                    outputs[output_idx] = x
                    output_idx += 1

            return outputs

        elif self.solver == 'rk4':
            num_steps = int(self.final_t / self.step_size)
            sampling_interval_steps = int(self.sampling_interval // self.step_size)
            num_outputs = (num_steps // sampling_interval_steps) + 1

            outputs = torch.zeros((int(num_outputs), num_nodes, x.size(1)), device=x.device, requires_grad=False)
            outputs[0] = x

            output_idx = 1
            for t_step in range(1, num_steps + 1):
                # Compute an RK4 step
                k1 = self.step_size * self.derivative_func(x, edge_index, batch)
                k2 = self.step_size * self.derivative_func(x + 0.5 * k1, edge_index, batch)
                k3 = self.step_size * self.derivative_func(x + 0.5 * k2, edge_index, batch)
                k4 = self.step_size * self.derivative_func(x + k3, edge_index, batch)
                
                x = x + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
                
                if t_step % sampling_interval_steps == 0:
                    outputs[output_idx] = x
                    output_idx += 1

            return outputs

    def message(self, x_j, norm):
        if norm is None:
            return x_j
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

if __name__ == '__main__':
    # Create dummy input data
    x = torch.randn(10, 3)  # Shape: [num_nodes, num_features]
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)  # Shape: [2, num_edges]
    # make edge_index undirected
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)  # Shape: [num_nodes]

    # Create an instance of PDE_layer
    pde_layer = PDE_layer()

    # Perform the forward pass
    solution = pde_layer.forward(x, edge_index, batch)

    # Print the shape of the solution
    print(solution.shape)
