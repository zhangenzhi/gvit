#!/bin/bash
# Begin LSF directives
#SBATCH -A gen006
#SBATCH -J Imagenet-ViT
#SBATCH -o Imagenet-ViT.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

#export PATH=$PWD/conda/bin:$PATH
#eval "$($PWD/conda/bin/conda shell.bash hook)"
set +x
conda activate /ccs/home/enzhi/miniconda_frontier/envs/pth

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/5.4.0

# grab nodecount
#nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
#nnodes=${#nodes[@]}

srun python -u test.py

# srun -n 4 --ntasks-per-node=4 -c 7 python -u cls_train.py

