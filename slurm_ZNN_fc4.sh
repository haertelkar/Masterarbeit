#!/bin/bash
#SBATCH --exclude=pool32
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -J "ZNN_fc4"
#SBATCH --mail-user=haertelk@physik.hu-berlin.de
#SBATCH --time=47:00:00
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py \
    -v 0307_1018_Z_TrE_4to8sparse_nz40_new_hi512_en3_fc4 \
    -e 9000 -nA 9 -nz 40 -m DQN -l labels_only_Dist.csv \
    -fa _4to8s_-50def_15B_new_40Z -hi 512 -enNL 3  -fcNL 4
