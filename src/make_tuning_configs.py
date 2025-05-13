"""
Creates config.yml and job.txt files for each combination of parameters for tuning.

***
May 2023

"""
import yaml
from pathlib import Path
import os
import itertools

pkg_path = '/***/***/pi/***/***/Graph_expressivity/'
root_path = 'tuning/'
python_path = '/***/***/pi/***/***/.conda_envs/pyg/bin/python'
# Define parameter ranges
params = {
    'pde_type': ['heat', 'wave'],
    'time_points': [10, 40],
    'time_range_start': [0.0],
    'time_range': [4.0, 8.0],
    'num_pde_layers': [2],
    'num_lin_layers_between': [1],
    'num_lin_layers_after': [1],
    'hidden_units': [128, 64, 256],
    'learning_rate': [1e-3, 1e-4],
    'weight_decay': [0.0],
    'p_dropout': [0.5, 0.0, 0.2],
    'skip_conn': [True, False],
    'batch_norm': [True, False],
    'data': ['ENZYMES'],
    'root_path': [root_path]
}

# Generate all combinations of parameters
keys, values = zip(*params.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

with open(f'{pkg_path}/{root_path}/job.txt', 'w') as job_file:
    # Write each combination to a YAML file
    for i, combination in enumerate(combinations):
        combination['config_id'] = i
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