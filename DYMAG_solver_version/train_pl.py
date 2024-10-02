import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from data_module import RandomDataset
from PDE_layer import heat_derivative_func
from DYMAG_pl import DYMAG_pl
import wandb

# Initialize wandb
wandb.init(project='DYMAG_project')

# Set a seed for reproducibility
torch.manual_seed(0)

# Set up the dataset
num_graphs = 25
num_graphs_test = 5
num_graphs_validation = 5
n_nodes = 10
graph_params_bounds = {'n': (n_nodes, n_nodes), 'p': (0.1, 0.5)}

train_dataset = RandomDataset(random_graph_model='er', num_graphs=num_graphs, graph_params_bounds=graph_params_bounds)
dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

test_dataset = RandomDataset(random_graph_model='er', num_graphs=num_graphs_test, graph_params_bounds=graph_params_bounds)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

validation_dataset = RandomDataset(random_graph_model='er', num_graphs=num_graphs_validation, graph_params_bounds=graph_params_bounds)
validation_dataloader = DataLoader(validation_dataset, batch_size=5, shuffle=False)


# Initialize the model
model = DYMAG_pl(input_feature_dim=n_nodes, output_dim=1, derivative_func=heat_derivative_func)

# Define the trainer
trainer = pl.Trainer(
    max_epochs=100,
    logger=pl.loggers.WandbLogger(),
    )

# Fit the model
trainer.fit(model, dataloader, validation_dataloader)

trainer.test(model, test_dataloader)

# Close the wandb run
wandb.finish()
