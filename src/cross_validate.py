"""
For cross-validation for the experiments
***
May 2023
"""
import sys
sys.path.insert(0, '/***/***/pi/***/***/Graph_expressivity/')
import argparse
import pickle
import torch
# from .classifier import GraphClassifierCustomizable
from experiments.classifier2 import PDEClassifier
from experiments.train import cross_validate_given_id_folds
from pathlib import Path
import numpy as np
import yaml

def eval_cv(args, data_path, result_path, device):
    config_id = args.config_id
    pde_type = args.pde_type
    time_points = args.time_points
    time_range_start = args.time_range_start
    time_range = args.time_range
    num_pde_layers = args.num_pde_layers
    num_lin_layers_between = args.num_lin_layers_between
    num_lin_layers_after = args.num_lin_layers_after
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    p_dropout = args.p_dropout
    skip_conn = args.skip_conn
    batch_norm = args.batch_norm
    ## did this to be compatible with old config.yml where batch_size and num_epochs aren't specified.
    batch_size = args.batch_size if args.batch_size is not None else 2048
    num_epochs = args.num_epochs if args.num_epochs is not None else 100
    with open(f"{data_path}/id_folds.pkl", "rb") as f:
        id_folds = pickle.load(f)
    with open(f"{data_path}/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    n_input = dataset[0].x.size(1)
    n_output = len(np.unique([data.y.item() for data in dataset]))
    ts = torch.linspace(time_range_start, time_range, time_points, dtype=torch.float, device=device)
    # model = GraphClassifierCustomizable(
    model = PDEClassifier(
        pde=pde_type, 
        ts=ts, 
        n_input=n_input, 
        n_hidden=hidden_units, 
        n_output=n_output, 
        device=device,
        num_layers=num_pde_layers,
        num_lin_layers_between_pde=num_lin_layers_between,
        num_lin_layers_after_pde=num_lin_layers_after,
        p_dropout=p_dropout,
        skip_conn=skip_conn,
        batch_norm=batch_norm
    ).to(device)
    # CV
    results_df, training_log_dict = cross_validate_given_id_folds(
        model, 
        dataset, 
        id_folds=id_folds, 
        batch_size=batch_size,  
        num_epochs=num_epochs,
        lr=learning_rate, 
        weight_decay=weight_decay, 
        device=device
    )
    # results_dir = Path(result_path)
    # results_dir.mkdir(parents=True, exist_ok=True)
    # Create a Path object for the config directory
    config_dir = Path(result_path) / str(config_id)
    # Ensure the config directory exists
    config_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config_dir / f'metrics.csv', index=False)
    save_training_log(result_path, config_id, training_log_dict)

def save_training_log(result_path, config_id, epoch_data):
    # Create a Path object for the config directory
    config_dir = Path(result_path) / str(config_id)
    # Ensure the config directory exists
    config_dir.mkdir(parents=True, exist_ok=True)
    # Iterate over the folds in epoch_data
    for fold, fold_epoch_data in epoch_data.items():
        # Create a filename for this fold's data
        fold_filename = config_dir / f'training_log_fold_{fold}.csv'
        # Save the DataFrame to a CSV file
        fold_epoch_data.to_csv(fold_filename, index=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int)
    parser.add_argument("--pde_type", type=str) 
    parser.add_argument("--time_points", type=int)
    parser.add_argument("--time_range_start", type=float, default=0.)
    parser.add_argument("--time_range", type=float)
    parser.add_argument("--num_pde_layers", type=int)
    parser.add_argument("--num_lin_layers_between", type=int)
    parser.add_argument("--num_lin_layers_after", type=int)
    parser.add_argument("--hidden_units", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--p_dropout", type=float, default=0.5)
    parser.add_argument("--skip_conn", type=bool, default=False)
    parser.add_argument("--batch_norm", type=bool, default=False)
    parser.add_argument("--data", type=str)
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=100)
    return parser

def save_config_to_yaml(args, yaml_file):
    with open(yaml_file, 'w') as f:
        yaml.dump(vars(args), f)

def load_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

if __name__ == "__main__":
    # root_path = '/***/***/pi/***/***/Graph_expressivity'

    parser = get_parser()
    args = parser.parse_args()

    if args.config_file:
        args = load_config_from_yaml(args.config_file)
        print(f'loading configuration from file <{args.config_id}>')
    else:
        save_config_to_yaml(args, f'config_{args.config_id}.yml')

    root_path = args.root_path
    data_path = args.data_path if args.data_path is not None else f"{root_path}/data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = f"{data_path}/{args.data}"
    result_path = f"{root_path}/{args.data}"
    eval_cv(args, data_path, result_path, device)
