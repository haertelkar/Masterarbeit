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
#SBATCH --array=1-4


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
srun python3 lightningTrain.py -v 3108_1328_onlyDistWeightedClip_atom$SLURM_ARRAY_TASK_ID -e 10 -m FullPixelGridML -l labels_only_Dist_$SLURM_ARRAY_TASK_ID.csv
