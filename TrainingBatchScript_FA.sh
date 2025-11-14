#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --partition=defq
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1            
#SBATCH --cpus-per-task=10
#SBATCH -J "4to15sDaten"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address 
#SBATCH --time=6-23:00:00


   


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Debugging flags (optional)
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script#
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py -v 1011_0955_Z_TrE_90OSA_4to15sA4to12s_hS512_enL5_0to150d_noH4 -e 9000 -nA 9 -m TrE -nz 90 -l labels_only_Dist.csv -fa _0411_4to12s_0to150def_15B_230Z_OSA,_0811_4to15s_0to150def_15B_230Z_OSA -enNL 5 -hi 512 -noH 4
