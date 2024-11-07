#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "tileSimulation"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address
#SBATCH --array=1-30

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python3 SimulateTilesOneFile.py -it 30 -id $SLURM_ARRAY_TASK_ID$RANDOM 
echo "finished tileSimulate.sh$SLURM_ARRAY_TASK_ID"
