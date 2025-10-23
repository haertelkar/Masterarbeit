#!/bin/bash
#Submit this script with: sbatch thefilename
#SBATCH --exclude pool32

#SBATCH --cpus-per-task=3   # number of processor cores 
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "tileSimulation"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address
#SBATCH --array=1-50
#SBATCH --time=47:00:00
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/data/scratch/haertelk/Masterarbeit/python-venv/bin/python SimulateTilesOneFile.py -it 200 -id $SLURM_ARRAY_TASK_ID$RANDOM 
echo "finished tileSimulate.sh$SLURM_ARRAY_TASK_ID"
