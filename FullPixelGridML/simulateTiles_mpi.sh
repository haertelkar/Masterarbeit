#!/bin/bash
#SBATCH --exclude pool32

#Submit this script with: sbatch thefilename

#SBATCH --time=48:00:00   # walltime
#SBATCH -J "tileSimulation"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mpirun -np 4 /data/scratch/haertelk/Masterarbeit/python-venv/bin/python SimulateTilesOneFile.py -it 80 -id $SLURM_ARRAY_TASK_ID$RANDOM 
echo "finished tileSimulate.sh$SLURM_ARRAY_TASK_ID"
