#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --partition=defq
#SBATCH --time=165:00:00
#SBATCH -J "ZNN"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python3 -u lightningTrain.py -v 0711_chamfer_1e-9_BatS4096_AdamW_1343 -e 800 -m DQN -l labels_only_Dist.csv
