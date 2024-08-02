import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch.nn as nn

def heat_derivative_func(lx):
    return -lx

class PDE_layer(MessagePassing):
    def __init__(self, num_nodes, derivative_func):
        super(PDE_layer, self).__init__(aggr='add')
        self.num_nodes = num_nodes
        self.step_size = .01
        self.solver = 'rk4'
        self.sampling_interval = .2
        self.final_t = 5
        self.derivative_func = derivative_func

        self.lin = nn.Linear(num_nodes, num_nodes)

    def get_laplacian(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        row, col = edge_index

        deg = degree(row, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def forward(self, x, edge_index):

        if self.solver == 'euler':
            num_steps = int(self.final_t / self.step_size)
            sampling_interval_steps = self.sampling_interval // self.step_size
            num_outputs = (num_steps // sampling_interval_steps) + 1

            outputs = torch.zeros((int(num_outputs), *x.shape), device = x.device, requires_grad=False)
            outputs[0] = x

            output_idx = 1
            for t_step in range(1, num_steps + 1):
                lx = self.get_laplacian(x, edge_index)
                dt = self.derivative_func(lx)
                x = x + self.step_size * dt
                if t_step % sampling_interval_steps == 0:
                    outputs[output_idx] = x
                    output_idx += 1

            return outputs

        elif self.solver == 'rk4':
            num_steps = int(self.final_t / self.step_size)
            sampling_interval_steps = self.sampling_interval // self.step_size
            num_outputs = (num_steps // sampling_interval_steps) + 1

            outputs = torch.zeros((int(num_outputs), *x.shape), device=x.device, requires_grad=False)
            outputs[0] = x

            output_idx = 1
            for t_step in range(1, num_steps + 1):
                # Compute an RK4 step
                # ref: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
                k1 = self.step_size * self.derivative_func(self.get_laplacian(x, edge_index))
                k2 = self.step_size * self.derivative_func(self.get_laplacian(x + 0.5 * k1, edge_index))
                k3 = self.step_size * self.derivative_func(self.get_laplacian(x + 0.5 * k2, edge_index))
                k4 = self.step_size * self.derivative_func(self.get_laplacian(x + k3, edge_index))
                
                x = x + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
                
                if t_step % sampling_interval_steps == 0:
                    outputs[output_idx] = x
                    output_idx += 1

            return outputs

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


if __name__ == '__main__':
    num_nodes = 10
    x = torch.randn(num_nodes,5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)

    model = PDE_layer(num_nodes, heat_derivative_func)
    output = model(x, edge_index)
    import pdb; pdb.set_trace()
    print(output)
    