import tphate
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from PDE_layer import PDE_layer
import phate 

if __name__ == '__main__':
    # get a random integer seed
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create an ER graph
    n_nodes = 25 
    p = 0.2
    num_traj = 5
    dynamics = 'sprott'
    if dynamics == 'sprott':
        phate_knn = 5
        final_t = 100
        sampling_interval = .5
    else:
        phate_knn = 20 
        final_t = 7
        sampling_interval = 0.1

    visualization = 'phate'
    # print out dynamics and visualization and other parameters
    print(f'Dynamics: {dynamics}')
    print(f'Visualization: {visualization}')
    print(f'Number of nodes: {n_nodes}')
    print(f'Number of trajectories: {num_traj}')
    print(f'Final time: {final_t}')
    print(f'Sampling interval: {sampling_interval}')

    G = nx.erdos_renyi_graph(n_nodes, p)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    x = torch.randn(n_nodes, num_traj) * 10
    # zero center the data
    x = x - x.mean(dim=0)
    batch = torch.tensor([0 for _ in range(n_nodes)], dtype=torch.long)

    # Create a PDE_layer instance
    pde_layer = PDE_layer(dynamics=dynamics, n_largest_graph = n_nodes,sampling_interval = sampling_interval, final_t = final_t, step_size = 0.01)

    # Perform forward pass
    outputs = pde_layer(x, edge_index, batch)
    # outputs has shape (num_steps, num_nodes, num_features)
    # plot the outputs at node 0 across all time steps for each trajectory
    # do this in the same plot
    for traj_ind in range(num_traj):
        plt.plot(outputs[:, 0, traj_ind], label=f'Trajectory {traj_ind}')
    plt.legend()
    # set the x-axis label to be the time step
    plt.xlabel('Forward recorded step')
    plt.ylabel('Node 0 value')
    plt.title('dynamics: ' + dynamics)
    # save the figure 
    plt.savefig(f'figs/outputs_plot_{dynamics}_{final_t}_{sampling_interval}_{phate_knn}_{seed}.png')
    # close 
    plt.close()



    if visualization == 'tphate':
        # outputs has shape (num_steps, num_nodes, num_features)
        # rearrange to (num_features, num_steps, num_nodes)
        outputs = outputs.permute(2, 0, 1).detach().numpy()
        import pdb; pdb.set_trace()
        # loop through each trajectory
        for traj_ind in range(num_traj):
            tphate_op = tphate.TPHATE(n_components=3, n_jobs=-1, verbose=0)
            data_tphate = tphate_op.fit_transform(outputs[traj_ind])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_tphate[:, 0], data_tphate[:, 1], data_tphate[:, 2])
            plt.show()
    elif visualization == 'phate':
        # outputs has shape (num_steps, num_nodes, num_features)
        total_n_steps = outputs.shape[0]
        # want to sparsify outputs at the end steps if dynamics are heat
        if dynamics == 'heat':
            outputs_time_mask = torch.ones(total_n_steps, dtype=int)
            midpt = total_n_steps // 2
            for i in range(midpt, total_n_steps):
                if i % 5 == 0:
                    outputs_time_mask[i] = 0
            for i in range(midpt + midpt//2 - 4, total_n_steps):
                if np.random.rand() < 0.8:
                    outputs_time_mask[i] = 0
            outputs = outputs[outputs_time_mask == 1]

        num_steps, num_nodes, num_features = outputs.shape
        print(f'num_steps: {num_steps}, num_nodes: {num_nodes}, num_features: {num_features}, total_n_steps: {total_n_steps}')
        # rearrange to (num_steps * num_features, num_nodes)
        outputs = outputs.permute(2,0,1).reshape(num_features * num_steps, num_nodes).detach().numpy()
        
        # this is stuipd but i'm lazy
        time_tracker = torch.zeros((num_steps, num_nodes, num_features), dtype=int)
        # set each value to the time step index
        for i in range(num_steps):
            time_tracker[i] = i
        time_tracker = time_tracker.permute(2,0,1).reshape(num_features * num_steps, num_nodes).detach().numpy()
        time_tracker = time_tracker[:,0]
        # make something to track num_features 
        traj_tracker = torch.zeros((num_steps, num_nodes, num_features), dtype=int)
        # set each value to the num_features index
        for i in range(num_features):
            traj_tracker[:,:,i] = i
        traj_tracker = traj_tracker.permute(2,0,1).reshape(num_features * num_steps, num_nodes).detach().numpy()
        traj_tracker = traj_tracker[:,0]
        
        # # Perform PHATE
        phate_op = phate.PHATE(n_components=3, n_jobs=-1, verbose=1, knn = phate_knn)
        data_phate = phate_op.fit_transform(outputs)

        # # Plot the results
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # color by the original time step
        # ax.scatter(data_phate[:, 0], data_phate[:, 1], data_phate[:, 2], c=time_tracker)
        # # add color bar
        # #ax.scatter(data_phate[:, 0], data_phate[:, 1], data_phate[:, 2])
        # plt.show()


        # Plot the results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define a colormap
        colormap = plt.cm.viridis

        # Normalize time_tracker for color mapping
        norm = plt.Normalize(vmin=time_tracker.min(), vmax=time_tracker.max())

        # Marker list
        marker_list = ['o', 'x', 's', 'D', '^']

        # Plot each trajectory with different markers and colors
        for traj_ind in range(num_features):
            traj_mask = traj_tracker == traj_ind
            colors = colormap(norm(time_tracker[traj_mask]))
            ax.scatter(data_phate[traj_mask, 0], data_phate[traj_mask, 1], data_phate[traj_mask, 2], 
                    label=f'IC {traj_ind}', c=colors, marker=marker_list[traj_ind])

        # Add legend
        plt.legend()

        # Add a color bar for the time
        mappable = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        mappable.set_array(time_tracker)
        cbar = plt.colorbar(mappable, ax=ax)
        # turn off color bar ticks
        cbar.set_ticks([])
        cbar.set_label('Time')
        # turn off axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.title(f'PHATE of {dynamics} dynamics')

        # Save the plot
        plt.savefig(f'figs/phate_plot_{dynamics}_{final_t}_{sampling_interval}_{phate_knn}_{seed}.png')
        # Show the plot
        plt.show()
        plt.close()
        print(f'saved to figs/phate_plot_{dynamics}_{final_t}_{sampling_interval}_{phate_knn}_{seed}.png')

        # Plot the results but color by the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        color_dict = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm'}

        # Marker list
        marker_list = ['o', 'x', 's', 'D', '^']

        # Plot each trajectory with different markers and colors
        for traj_ind in range(num_features):
            traj_mask = traj_tracker == traj_ind
            ax.scatter(data_phate[traj_mask, 0], data_phate[traj_mask, 1], data_phate[traj_mask, 2], 
                    label=f'IC {traj_ind}', c=color_dict[traj_ind], marker=marker_list[traj_ind])

        # Add legend
        plt.legend()


        # turn off axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.title(f'PHATE of {dynamics} dynamics')

        # Save the plot
        plt.savefig(f'figs/phate_plot_traj_color_{dynamics}_{final_t}_{sampling_interval}_{phate_knn}_{seed}.png')
        # Show the plot
        plt.show()
