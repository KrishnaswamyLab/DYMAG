from experiments.sol_propagator import PDESolutionPropogator
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np



def plot_pde_sols(graphs, device, c=.5, dirac_ids=[0], names=None, random_dirac=False, random_num=1, display_labels=False):
    """
    Compute the heat/wave eqn solutions and plot.
    for wave eqation, the initial signal (x) and velocity (y) are taken to be the same.

    Args:
        graphs (list): list of networkx graphs
        device (device): torch device. eqn solving can be on gpu.
        c (float, optional): heat material/wave speed in the eqns. Defaults to .5.
        dirac_ids (list, optional): indices where to put the dirac. Defaults to 0.
        names (list, optional): list of names of graphs. Defaults to None. will use number.
        random_dirac (bool, optional): if True dirac_ids is not used, and will randomly generate ids. Defaults to False.
        random_num (int, optional): how many diracs to put if randomly generate. Defaults to 1.
        display_labels (bool, optional): Whether or not display labels in graph plot. Defaults to False.

    Returns:
        dictionary(name: dictionary('heat/wave':result matrix))
    """
    if names is None: names = [i for i in range(len(graphs))]
    figsize_inches = 3
    fig, axes = plt.subplots(len(graphs), 3, figsize=(figsize_inches*3, figsize_inches*len(graphs)), dpi=250)
    res = dict()
    for graph_idx in range(len(graphs)):
        graph = graphs[graph_idx]
        laplacian_mat = nx.normalized_laplacian_matrix(graph).tocsc()
        oper = PDESolutionPropogator(laplacian_mat, device, eps=1e-5)
        p = torch.empty(graph.number_of_nodes()).uniform_(0, 0.2)
        x_init = torch.zeros(graph.number_of_nodes(), dtype=torch.float, device=device)
        if random_dirac:
            dirac_ids = np.random.choice(graph.number_of_nodes(), size=random_num, replace=False)
        x_init[dirac_ids] = 1. # Dirac delta

        x = x_init.view(-1, 1, 1).to(device)

        ts = torch.linspace(0, 10, 100).float().to(device)
        yHeat = oper.propogateHeat(x, ts, c=c)
        yWave = oper.propogateWave(x, x, ts, c=c)

        colors = ['red' if node in dirac_ids else 'darkgray' for node in graph.nodes]

        nx.draw_networkx(graph, ax=axes[graph_idx, 0], edge_color="lightgray", alpha=0.75,
                        node_size=10, node_color=colors, with_labels=display_labels)
        axes[graph_idx, 0].set_title(f"{names[graph_idx]}")

        # Heat subplot
        im1 = axes[graph_idx, 1].imshow(yHeat.squeeze().cpu().numpy().T, aspect='auto')
        fig.colorbar(im1, ax=axes[graph_idx, 1])
        axes[graph_idx, 1].set_xlabel("Time")
        axes[graph_idx, 1].set_ylabel("Nodes")
        axes[graph_idx, 1].set_title("Heat")

        # Wave subplot
        im2 = axes[graph_idx, 2].imshow(yWave.squeeze().cpu().numpy().T, aspect='auto')
        fig.colorbar(im2, ax=axes[graph_idx, 2])
        axes[graph_idx, 2].set_xlabel("Time")
        axes[graph_idx, 2].set_ylabel("Nodes")
        axes[graph_idx, 2].set_title("Wave")

        # plt.suptitle(f"Graph {graph_idx}")

        # Display the subplots
        plt.tight_layout()
        res_this = {'heat': yHeat.squeeze().cpu().numpy().T, 'wave': yWave.squeeze().cpu().numpy().T}
        res[names[graph_idx]] = res_this
    plt.show()
    return res

def get_pde_sol(graph, ts, dirac_ids, device, c=0.5):
    """_summary_

    Args:
        graph (_type_): _description_
        ts (_type_): _description_
        dirac_ids (_type_): _description_
        c (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    laplacian_mat = nx.normalized_laplacian_matrix(graph).tocsc()
    oper = PDESolutionPropogator(laplacian_mat, device, eps=1e-5)
    x_init = torch.zeros(graph.number_of_nodes(), dtype=torch.float, device=device)
    x_init[dirac_ids] = 1. # Dirac delta
    x = x_init.view(-1, 1, 1).to(device)
    yHeat = oper.propogateHeat(x, ts, c=c)
    yWave = oper.propogateWave(x, x, ts, c=c)
    return yHeat.squeeze().cpu().numpy().T, yWave.squeeze().cpu().numpy().T