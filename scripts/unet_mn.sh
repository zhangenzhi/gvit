#!/bin/bash
#SBATCH -A bif146
#SBATCH -o unetr.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_LOCALID
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=29500 # default from torch launcher

set +x
source /lustre/orion/bif146/world-shared/gvit/env/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/bif146/world-shared/gvit/env/miniconda3/env/gvit


# module load PrgEnv-gnu
# module load gcc/12.2.0
module load rocm/5.7.0

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

# exec
srun -n 2 --ntasks-per-node 8 -c 7 \
    python3 unet_mn.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=1024 \
        --epoch=100 \
        --batch_size=4 \
        --savefile=./unet-mn-1k
