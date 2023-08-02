#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=16               
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00

# Debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python3 lightningTrain.py -v 11x11CTuner150 -e 150 -i 4,7 -m omplex