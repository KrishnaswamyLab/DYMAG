import nolds
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from PDE_layer import PDE_layer
import phate 
import time
import argparse
from data_module import RandomDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import pandas as pd
import os
import multiprocessing as mp
import numpy as np

def compute_lyapunov_for_node(traj):
    # Calculate the maximum Lyapunov exponent for a given trajectory
    return nolds.lyap_r(traj)

def compute_lyapunov_for_feature(x, num_nodes, traj_ind):
    lyap_max = 0
    # Calculate Lyapunov exponents for each node in the feature
    for node_ind in range(num_nodes):
        traj = x[:, node_ind, traj_ind]
        l = compute_lyapunov_for_node(traj)
        if l > lyap_max:
            lyap_max = l
    return lyap_max

def measure_lyapunov(x):
    # Assume that x is outputs
    x = x.detach().cpu().numpy()
    num_steps, n_nodes, num_features = x.shape

    # Prepare arguments for parallel processing
    args = [(x, n_nodes, traj_ind) for traj_ind in range(num_features)]
    
    # Use multiprocessing to parallelize across features
    with mp.Pool() as pool:
        lyap_max_list = pool.starmap(compute_lyapunov_for_feature, args)

    return lyap_max_list

def get_time_diffs(x1, x2):
    # x1 and x2 have shape (num_steps, n_nodes, num_features)
    diff = x1-x2
    # want the norm to be matrix norm, resulting in final shape of num_steps
    norm = np.linalg.norm(diff, axis=(1,2))
    return norm

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_nodes', type=int, default=25, help='Number of nodes')
    argparser.add_argument('--num_reps_per_n_node', type=int, default=5, help='Number of repetitions per number of nodes')
    argparser.add_argument('--num_graphs', type=int, default=10, help='Number of graphs')
    argparser.add_argument('--signal_type', type=str, default='dirac', help='Type of signal to use')
    argparser.add_argument('--sampling_interval', type=float, default=0.2, help='Sampling interval')
    argparser.add_argument('--final_t', type=float, default=20, help='Final time')
    argparser.add_argument('--step_size', type=float, default=0.01, help='Step size')
    argparser.add_argument('--edge_addition', type=int, default=2, help='Number of edges to add to the graph')
    args = argparser.parse_args()

    # generate one graph
    n_nodes = args.n_nodes
    seed = 24
    torch.manual_seed(seed)
    np.random.seed(seed)
    signal_type = args.signal_type

    graph_params_bounds = {'n': (n_nodes, n_nodes), 'p': (.1, .5)}
    # get dataset
    data_unperturbed = RandomDataset(random_graph_model='er', 
                            num_graphs=1, 
                            graph_params_bounds=graph_params_bounds, 
                            node_features=signal_type)
    
    # create a data object with the same graph but add a random edge
    G = data_unperturbed[0]
    data_unperturbed = data_unperturbed[0]
    edge_index = G.edge_index
    edge_addition = args.edge_addition
    added_edges = 0
    while added_edges < edge_addition:
        node0, node1 = torch.randint(0, n_nodes, (2,))
        edge_in_index = (node0 == edge_index[0]) & (node1 == edge_index[1])
        if edge_in_index.any():
            continue
        edge_index = torch.cat([edge_index, torch.tensor([[node0, node1], [node1, node0]])], dim=1)
        print(f'added edge between {node0} and {node1}')
        added_edges += 1
    data_perturbed = Data(x=G.x, edge_index=edge_index)

    pde_layer_sprott = PDE_layer(dynamics='sprott', n_largest_graph = n_nodes,sampling_interval = args.sampling_interval, final_t = args.final_t, step_size = args.step_size)
    pde_layer_heat = PDE_layer(dynamics='heat', n_largest_graph = n_nodes,sampling_interval = args.sampling_interval, final_t = args.final_t, step_size = args.step_size)

    batch = torch.tensor([0 for _ in range(n_nodes)], dtype=torch.long)
    sprott_perturbed = pde_layer_sprott(data_perturbed.x, data_perturbed.edge_index, batch)
    sprott_unperturbed = pde_layer_sprott(data_unperturbed.x, data_perturbed.edge_index, batch)
    heat_perturbed = pde_layer_heat(data_perturbed.x, data_perturbed.edge_index, batch)
    heat_unperturbed = pde_layer_heat(data_unperturbed.x, data_perturbed.edge_index, batch)

    # get lyapunov exponents
    print('getting lyapunov for sprott perturbed')
    lyap_sprott_perturbed = measure_lyapunov(sprott_perturbed)
    print('getting lyapunov for sprott unperturbed')
    lyap_sprott_unperturbed = measure_lyapunov(sprott_unperturbed)
    print('getting lyapunov for heat perturbed')
    lyap_heat_perturbed = measure_lyapunov(heat_perturbed)
    print('getting lyapunov for heat unperturbed')
    lyap_heat_unperturbed = measure_lyapunov(heat_unperturbed)

    # output has shape (num_steps, n_nodes, num_features)
    deltas_sprott = get_time_diffs(sprott_perturbed, sprott_unperturbed)
    deltas_heat = get_time_diffs(heat_perturbed, heat_unperturbed)

    # print out deltas sprott and heat every 5 steps 
    print('sprott deltas every second:')
    deltas_sprott_sparse = deltas_sprott[::5]
    print(deltas_sprott_sparse)
    print('heat deltas every second:')
    deltas_heat_sparse = deltas_heat[::5]
    print(deltas_heat_sparse)


    # plot the results
    plt.figure()
    plt.plot(deltas_sprott, label='Sprott')
    plt.plot(deltas_heat, label='Heat')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Norm of difference')
    plt.title('Difference between perturbed and unperturbed graphs')
    plt.show()

    # make a 2x2 subplot and plot the solution over time at node 0 feature 0
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(sprott_perturbed[:, 0, 0])
    plt.title('Sprott perturbed')
    plt.subplot(2,2,2)
    plt.plot(sprott_unperturbed[:, 0, 0])
    plt.title('Sprott unperturbed')
    plt.subplot(2,2,3)
    plt.plot(heat_perturbed[:, 0, 0])
    plt.title('Heat perturbed')
    plt.subplot(2,2,4)
    plt.plot(heat_unperturbed[:,0,0])
    plt.title('Heat unperturbed')
    plt.show()

    # print out lyapunov exponents
    print('Lyapunov exponents:')
    print('Sprott perturbed:', lyap_sprott_perturbed)
    print('Sprott unperturbed:', lyap_sprott_unperturbed)
    print('Heat perturbed:', lyap_heat_perturbed)
    print('Heat unperturbed:', lyap_heat_unperturbed)
