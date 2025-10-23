#!/bin/bash
#SBATCH --exclude pool32

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "combineLabels"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python combineLabelsCSVs.py _1710_4to8s_-50def_15B_860Z_OSA

