#!/bin/bash
#SBATCH --exclude pool32

# SLURM SUBMIT SCRIPT
#SBATCH --partition=short
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=10    
#SBATCH --cpus-per-task=1
#SBATCH -J "100ZNN"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address   
#SBATCH --time=47:00:00      


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script#
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py -v 1705_0939_Z_TrE_14sparse_noEB_NoNoise_newScaling_-150def_40ZM_9000E -e 9000 -m TrE -l labels_only_Dist.csv -np 16 -nz 40 -lb 14 