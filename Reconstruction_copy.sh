#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=10    
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH -J "TestRecShort"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email addres


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python Reconstruction.py

