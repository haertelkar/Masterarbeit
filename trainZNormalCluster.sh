#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=10
#SBATCH -J "cTrainZNormal"   # job name
#SBATCH --mail-user=haertelk@physik.hu-berlin.de   # email address

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo $head_node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun python3 -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_endpoint $head_node_ip:29500 --rdzv_backend c10d trainCluster.py -v 11x11C -e 150 -i 4,7 -m ormal




# srun torchrun \
# --nnodes 4 \
# --nproc_per_node 1 \
# --rdzv_id $RANDOM \
# --rdzv_backend c10d \
# --rdzv_endpoint $head_node_ip:29500 \
# /shared/examples/multinode_torchrun.py 50 10