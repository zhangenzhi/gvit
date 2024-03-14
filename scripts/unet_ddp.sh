#!/bin/bash
#SBATCH -A bif146
#SBATCH -o unet.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

# export PATH=$PWD/conda/bin:$PATH
# eval "$($PWD/conda/bin/conda shell.bash hook)"

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

# set +x
# source /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/etc/profile.d/conda.sh
# conda activate /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/envs/gvit

# export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

# grab nodecount
# nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
# nnodes=${#nodes[@]}

# exec
srun -n 1 --ntasks-per-node=1 -c 1 python3 unet_train_ddp.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=512 \
        --epoch=100 \
        --batch_size=8 \
        --savefile=./unet_vis_ddp