#!/bin/bash
#SBATCH --exclude pool32

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH -J "TestRecShort"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email addres
#SBATCH --exclude pool33
#SBATCH --array=1-8


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python Reconstruction.py --makeNewFolder 2025_07_20_ViT_-50def_substract_emptyS --onlyPred --defocus -50 --sparseGridFactor $SLURM_ARRAY_TASK_ID -b 15 --pixel --model /data/scratch/haertelk/Masterarbeit/checkpoints/visionTransformer_1307_1634_viT_TrE_4to8sparse_hiddenSize512_-50def_s1/epoch=44-step=303750.ckpt
