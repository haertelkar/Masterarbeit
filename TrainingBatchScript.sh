#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --partition=defq
#SBATCH --time=165:00:00
#SBATCH -J "ZNN"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address
#SBATCH -w pool2

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python3 -u lightningTrain.py -v 1212_1330_Z_GRU_bReshape_RandPosWithCoords_13000E -e 13000 -m DQN -l labels_only_Dist.csv
