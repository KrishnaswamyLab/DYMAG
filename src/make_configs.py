"""
Creates config.yml and job.txt files for running experiments.

Xingzhi Sun
May 2023

"""
import yaml
from pathlib import Path
import os
import itertools

pkg_path = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Graph_expressivity/'
root_path = 'classification_experiments/'
python_path = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/pyg/bin/python'
# Define parameter ranges
params = {
    'pde_type': ['heat', 'wave'],
    'time_points': [10],
    'time_range_start': [0.0],
    'time_range': [4.0],
    'num_pde_layers': [2],
    'num_lin_layers_between': [1],
    'num_lin_layers_after': [1],
    'hidden_units': [256],
    'learning_rate': [1e-4],
    'weight_decay': [0.0],
    'p_dropout': [0.5],
    'skip_conn': [False],
    'batch_norm': [True],
    'data': ['ENZYMES', 'PROTEINS', 'MUTAG'],
    'root_path': [root_path],
    'data_path': ['data/'],
    'batch_size': [2048],
    'num_epochs': [1000]
}

# Generate all combinations of parameters
keys, values = zip(*params.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

dir_path = f"{pkg_path}/{root_path}"
Path(dir_path).mkdir(exist_ok=True, parents=True)

config_id_start = 36 ## can specify the starting config_id here.
with open(f'{pkg_path}/{root_path}/job.txt', 'w') as job_file:
    # Write each combination to a YAML file
    for i, combination in enumerate(combinations):
        combination['config_id'] = i + config_id_start
        config_id = combination['config_id']
        data = combination['data']
        dir_path = f"{pkg_path}/{root_path}/{data}/{config_id}"
        Path(dir_path).mkdir(exist_ok=True, parents=True)
        with open(f"{dir_path}/config.yml", 'w') as f:
            yaml.dump(combination, f)

        job_file.write(f"cd {pkg_path}; {python_path} src/cross_validate.py --config_file {root_path}/{data}/{config_id}/config.yml\n")

# Write run_job.sh to run all jobs
with open(f'{pkg_path}/{root_path}/run_job.sh', 'w') as dsq_file:
    dsq_file.write("""#!/bin/sh
module load dSQ;
filename=job;
dSQ --jobfile ${filename}.txt --mem-per-gpu=40G -t 1-01:00:00 -n 1 -p gpu,scavenge_gpu --gpus=1 -J ${filename}
sbatch dsq-${filename}-$(date +%Y-%m-%d).sh
""")