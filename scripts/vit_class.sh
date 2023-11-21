# export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)


# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID
# export WORLD_SIZE=$SLURM_LOCALID
# export MASTER_ADDR=$HOSTNAME
# export MASTER_PORT=29500 # default from torch launcher
# export BATCH_DDP=1


export VALUE=1

OMP_NUM_THREADS=$VALUE torchrun --nnodes=1 --nproc_per_node=8  ./main.py
