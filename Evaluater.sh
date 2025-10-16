#!/bin/bash
#SBATCH --exclude pool32

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH -J "Evaluater"   # job name

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1



# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun /data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u  evaluateCHKP.py /data/scratch/haertelk/Masterarbeit/checkpoints/DQN_1506_1254_Z_TrE_4to7sparse_nz16_cont2_-50def_s1