#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1          
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH -J "ZNN"   # job name
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python3 lightningTrain.py -v 1905_onlyGridPos -e 25 -m ormal 
