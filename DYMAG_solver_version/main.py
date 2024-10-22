from data_module import RandomDataset
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import networkx as nx
from PDE_layer import PDE_layer
from DYMAG import DYMAG
import argparse 

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Run DYMAG on a RandomDataset of Erdos-Renyi graphs')
    args.add_argument('--num_graphs', type=int, default=25, help='Number of training graphs to generate')
    args.add_argument('--num_graphs_test', type=int, default=5, help='Number of test graphs to generate')
    args.add_argument('--n_nodes', type=int, default=100, help='Number of nodes in each graph')
    args.add_argument('--p_min', type=float, default=0.1, help='Minimum edge probability')
    args.add_argument('--p_max', type=float, default=0.5, help='Maximum edge probability')
    args.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda)')
    args.add_argument('--dynamic', type=str, default='sprott', help='Dynamics to model')    
    args = args.parse_args()

    # run DYMAG on a RandomDataset of Erdos-Renyi graphs
    # set a seed for reproducability 
    torch.manual_seed(0)
    # set up the dataset
    num_graphs = args.num_graphs
    num_graphs_test = args.num_graphs_test
    n_nodes = args.n_nodes
    graph_params_bounds = {'n': (n_nodes, n_nodes), 'p': (args.p_min, args.p_max)}

    train_dataset = RandomDataset(random_graph_model='er', num_graphs=num_graphs, graph_params_bounds=graph_params_bounds)
    dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    test_dataset = RandomDataset(random_graph_model='er', num_graphs=num_graphs_test, graph_params_bounds=graph_params_bounds)

    model = DYMAG(input_feature_dim=n_nodes, output_dim=1, dynamics=args.dynamic, n_largest_graph=n_nodes, device=args.device)

    # set up the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    num_epochs = 100
    # training loop
    for epoch in range(num_epochs):
        total_loss = 0

        for data in dataloader:
            optimizer.zero_grad()
            
            # forward pass
            output = model(data.x, data.edge_index, data.batch)

            # compute loss
            loss = criterion(output, data.y)

            # backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

    # run DYMAG on the test dataset and compute the average loss
    total_loss = 0
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    for data in test_dataloader:
        x = data.x
        edge_index = data.edge_index
        batch_index = data.batch
        # forward pass
        output = model(x, edge_index, batch_index)
        loss = criterion(output, data.y)
        total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"Average Loss on Test Dataset: {avg_loss}")
