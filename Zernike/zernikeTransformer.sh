#!/bin/bash
#SBATCH --exclude pool32

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH --nodes=10   # number of nodes
#SBATCH -J "zernTransf"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mpirun /data/scratch/haertelk/Masterarbeit/python-venv/bin/python ZernikeTransformer.py