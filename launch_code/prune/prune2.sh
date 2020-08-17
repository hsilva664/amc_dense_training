#!/bin/bash

#SBATCH -p mmaire-gpu --exclude=gpu-g6 -c 1 -J h_amc_pru -d singleton --array=1-24

bash -c ". ~/torchenv/bin/activate; sh ~/begin.sh; `sed "${SLURM_ARRAY_TASK_ID}q;d" launch_code/prune/prune2.txt`"