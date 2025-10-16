#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --partition=defq
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1            
#SBATCH --cpus-per-task=10
#SBATCH -J "100ZNN"   # job name
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
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py -v 2007_1131_viT_TrE_4to8sparse_hiddenSize512_enL3_CSL_-50def -e 9000 -nA 9 -m visionTransformer -l labels_only_Dist.csv -fa _4to8s_-50def_15B_new -hi 512 -enNL 3
