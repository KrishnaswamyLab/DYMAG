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
import pandas as pd
import os
import multiprocessing as mp
import numpy as np

def compute_lyapunov_for_node(traj):
    # Calculate the maximum Lyapunov exponent for a given trajectory
    return max(nolds.lyap_e(traj))

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



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dynamics', type=str, default='sprott', help='Dynamics to simulate')
    argparser.add_argument('--n_node_list', type=int, nargs='+', default=[25, 50, 100, 317], help='List of number of nodes')
    argparser.add_argument('--num_reps_per_n_node', type=int, default=5, help='Number of repetitions per number of nodes')
    argparser.add_argument('--num_graphs', type=int, default=10, help='Number of graphs')
    argparser.add_argument('--signal_type', type=str, default='dirac', help='Type of signal to use')
    argparser.add_argument('--sampling_interval', type=float, default=0.2, help='Sampling interval')
    argparser.add_argument('--final_t', type=float, default=20, help='Final time')
    argparser.add_argument('--step_size', type=float, default=0.01, help='Step size')
    args = argparser.parse_args()

    results = pd.DataFrame(columns=['n_nodes','rep','graph_ind','lyap_mean', 'lyap_min', 'lyap_max', 'lyap_std', 'time'])
    for n_nodes in args.n_node_list:
        print('Number of nodes:', n_nodes)
        for rep in range(args.num_reps_per_n_node):
            print('Repetition:', rep)
            # get a random integer seed
            seed = rep
            torch.manual_seed(seed)
            np.random.seed(seed)

            graph_params_bounds = {'n': (n_nodes, n_nodes), 'p': (.1, .5)}
            # get dataset
            dataset = RandomDataset(random_graph_model='er', 
                                    num_graphs=args.num_graphs, 
                                    graph_params_bounds=graph_params_bounds, 
                                    node_features=args.signal_type)

            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            pde_layer = PDE_layer(dynamics=args.dynamics, n_largest_graph = n_nodes,sampling_interval = args.sampling_interval, final_t = args.final_t, step_size = args.step_size)

            for graph_ind, data in enumerate(dataloader):
                x = data.x
                edge_index = data.edge_index
                batch = data.batch
                start_time = time.time()
                outputs = pde_layer(x, edge_index, batch)
                time_elapsed = time.time() - start_time
                lyap_max_list = measure_lyapunov(outputs)
                lyap_mean = np.mean(lyap_max_list)
                lyap_min = np.min(lyap_max_list)
                lyap_max = np.max(lyap_max_list)
                lyap_std = np.std(lyap_max_list)
                new_row = {'n_nodes': n_nodes, 'rep': rep, 'graph_ind': graph_ind, 'lyap_mean': lyap_mean, 'lyap_min': lyap_min, 'lyap_max': lyap_max, 'lyap_std': lyap_std, 'time': time_elapsed}
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                print(f'Results for graph {graph_ind}:')
                print(f'Lyapunov mean: {lyap_mean}, Lyapunov min: {lyap_min}, Lyapunov max: {lyap_max}, Lyapunov std: {lyap_std}, Time: {time_elapsed}')
                
    save_string = f'lyapunov_results_{args.dynamics}_{args.signal_type}_{args.sampling_interval}_{args.final_t}_{args.step_size}.csv'
    results_dir = 'lyapunov_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_string = os.path.join(results_dir, save_string)
    results.to_csv(save_string, index=False)
