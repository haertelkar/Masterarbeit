#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --partition=defq
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=10    
#SBATCH --cpus-per-task=1
#SBATCH -J "100ZNN"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address         


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script#
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py -v 0105_1524_Z_TrE_49P_10sparse_EB_noCSL_20ZM_9000E -e 9000 -m DQN -l labels_only_Dist.csv -np 49 -nz 20 -fa _10sparse
