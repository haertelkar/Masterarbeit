#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=30:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=10   # number of nodes
#SBATCH -J "tileSimulation"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
for i in {1..10}; do
    python3 SimulateMoreComplexTiled.py -it 20 -id $i$RANDOM
done
