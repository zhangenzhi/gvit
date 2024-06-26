#!/bin/bash
#SBATCH -A bif146
#SBATCH -o vitunetr2d.o%J
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -p batch

# export PATH=$PWD/conda/bin:$PATH
# eval "$($PWD/conda/bin/conda shell.bash hook)"

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

# grab nodecount
# nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
# nnodes=${#nodes[@]}

# exec
srun -n 1 --ntasks-per-node=1 -c 1 python3 vitunetr2d_train.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=512 \
        --tokens=576 \
        --epoch=1000 \
        --batch_size=16 \
        --savefile=./vis_vitunet_512

srun -n 1 --ntasks-per-node=1 -c 1 python3 vitunetr2d_train.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=1024 \
        --tokens=1024 \
        --epoch=1000 \
        --batch_size=8 \
        --savefile=./vis_vitunet_1k

srun -n 1 --ntasks-per-node=1 -c 1 python3 vitunetr2d_train.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=4096 \
        --tokens=4096 \
        --epoch=1000 \
        --batch_size=1 \
        --savefile=./vis_vitunet_4k