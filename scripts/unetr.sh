#!/bin/bash
#SBATCH -A bif146
#SBATCH -o unetr.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

# set +x
# source ~/miniconda_frontier/etc/profile.d/conda.sh
# conda activate /ccs/home/enzhi/miniconda_frontier/envs/gvit

# export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

# exec
srun -n 1 --ntasks-per-node=1 -c 1 python3 unetr_train.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=1024 \
        --epoch=500 \
        --batch_size=2 \
        --savefile=./unetr_visual-1k