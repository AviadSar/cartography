#!/bin/bash

#SBATCH --mem=8g
#SBATCH --gres=gpu:rtx2080,vmem:10g
#SBATCH --time=8:0:0
#SBATCH --output=/cs/labs/roys/aviadsa/cartography/slurm_out_files/huji_winogrande_deberta_on_deberta_033.txt

python python cartography/classification/run_glue.py -c configs/huji_winogrande_deberta_on_deberta_033.jsonnet --do_train -o outputs/huji_winogrande_deberta_on_deberta_033/