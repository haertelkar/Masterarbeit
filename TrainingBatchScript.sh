#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --partition=defq
#SBATCH --time=165:00:00
#SBATCH -J "9ZNN"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address
#SBATCH --exclusive

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python3 -u lightningTrain.py -v 1312_1108_Z_GRU_onlyLastHid_9RandPosWithCoords_9000E -e 9000 -m DQN -l labels_only_Dist.csv -np 9
