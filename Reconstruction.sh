#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH -J "TestRecFull"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email addres

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need this:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Run your training script
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python Reconstruction.py --defocus -50 --sparseGridFactor 8 -b 15 --model /data/scratch/haertelk/Masterarbeit/checkpoints/DQN_3006_1041_Z_TrE_4to8sparse_nz40_new_hi512_en_fc_s1/epoch=27-step=55104.ckpt
./runAllROPs.sh
