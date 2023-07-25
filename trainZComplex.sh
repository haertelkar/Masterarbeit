#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "trainZComplex"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python3 train.py -v AdamWDenseOnlyClosest -e 150 -i 4,7 -m omplex