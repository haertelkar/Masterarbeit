#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH -J "YOLOTrainingGen"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email addres
#SBATCH --array=0-9


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python Yolo_Training_Data_Generator.py --structure $SLURM_ARRAY_TASK_ID --makeNewFolder YOLORUN_2025_07_06_8s_-50def --onlyPred --defocus -50 --sparseGridFactor 8 -b 15 --model /data/scratch/haertelk/Masterarbeit/checkpoints/DQN_3006_1041_Z_TrE_4to8sparse_nz40_new_hi512_en_fc_s1/epoch=27-step=55104.ckpt
