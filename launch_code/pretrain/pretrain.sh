#!/bin/bash

#SBATCH -p mmaire-gpu --exclude=gpu-g6 -c 1 -J h_amc_pre -d singleton --array=1-6

bash -c ". ~/torchenv/bin/activate; sh ~/begin.sh; `sed "${SLURM_ARRAY_TASK_ID}q;d" launch_code/pretrain/pretrain.txt`"