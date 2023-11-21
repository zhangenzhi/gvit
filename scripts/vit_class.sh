# export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

torchrun --nnodes=1 --nproc_per_node=8 main.py