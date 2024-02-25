#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH -J "ZNN"   # job name

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python3 lightningTrain.py -v OneFileAllPosFixNewResults -e 300 -m ormal 
