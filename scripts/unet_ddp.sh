#!/bin/bash
#SBATCH -A bif146
#SBATCH -o unet.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"
cp resnet18-f37072fd.pth /tmp/srun/.cache/torch/hub/checkpoints/

# export PATH="/lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/bin:$PATH"

# set +x
# source /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/etc/profile.d/conda.sh
# conda activate /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/envs/gvit

# cp 
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

# exec
srun -n 1 --ntasks-per-node=1 -c 1 python3 unet_train_ddp.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=1024 \
        --epoch=10 \
        --batch_size=8 \
        --savefile=./unet_vis_1k

# 8-gpu
python3 unet_train_ddp.py \
        -g=8 \
        -n=1 \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=1024 \
        --epoch=10 \
        --batch_size=8 \
        --savefile=./unet_8g