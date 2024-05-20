#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=20   # number of processor cores (i.e. tasks)
#SBATCH --nodes=20   # number of nodes
#SBATCH -J "tileSimulation"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
for i in {1..80}; do
    srun -N 1 -n 1 python3 SimulateTilesOneFile.py -it 20 -id $i$RANDOM &
done
wait
echo 'finished tileSimulate.sh'
