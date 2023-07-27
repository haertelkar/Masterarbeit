#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks-per-node=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=2   # number of nodes
#SBATCH -J "cTrainFullPixelGrid"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mpirun -n $SLURM_NTASKS python3 trainCluster.py -v AdamWDenseOnlyClosest2x2C -e 150 -i 4,7 -m ull
#srun python3 trainCluster.py --rank $SLURM_PROCID --world_size $SLURM_NTASKS -v AdamWDenseOnlyClosest2x2C -e 150 -i 4,7 -m ull
#srun python3 -u trainCluster.py --rank $SLURM_PROCID --world_size $SLURM_NTASKS -v AdamWDenseOnlyClosest2x2C -e 150 -i 4,7 -m ull
#python3  -v AdamWDenseOnlyClosest2x2 -e 150 -i 4,7 -m ull