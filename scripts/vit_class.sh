export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py