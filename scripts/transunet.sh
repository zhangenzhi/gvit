#!/bin/bash
#SBATCH -A bif146
#SBATCH -o transunet.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

#export PATH=$PWD/conda/bin:$PATH
#eval "$($PWD/conda/bin/conda shell.bash hook)"
set +x
source ~/miniconda_frontier/etc/profile.d/conda.sh
conda activate /ccs/home/enzhi/miniconda_frontier/envs/gvit

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/5.4.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# grab nodecount
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
nnodes=${#nodes[@]}

# exec
python3 transunet_train.py \
        --datapath=./dataset/paip/output_images_and_masks \
        --resolution=512 \
        --epoch=100 \
        --batch_size=16 \
        --savefile=./transunet_visual