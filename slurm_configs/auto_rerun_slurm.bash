#!/bin/bash
#SBATCH --time=48:0:0
#SBATCH --output=/cs/labs/roys/aviadsa/cartography/slurm_configs/auto_rerun_slurm.out

python slurm_configs/auto_rerun_slurm.py
