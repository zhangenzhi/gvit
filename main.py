import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
from model.vision_transformer import VisionTransformer
from torch.nn.parallel import DistributedDataParallel as DDP

# os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
# os.environ['MASTER_PORT'] = "29500"
# os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
# os.environ['RANK'] = os.environ['SLURM_PROCID']

# world_size = int(os.environ['SLURM_NTASKS'])
# world_rank = int(os.environ['SLURM_PROCID'])
# local_rank = int(os.environ['SLURM_LOCALID'])


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    model = VisionTransformer(
                image_size=256,
                patch_size=32,
                num_layers=12,
                num_heads=12,
                hidden_dim=3072,
                mlp_dim=3072,
                num_classes=1000)
    
    device_id = rank % torch.cuda.device_count()
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)

    loss_fn = nn.CrossEntropyLoss()
    # print(ddp_model.parameters())
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    BATCH_SIZE=4
    for i in range(1000):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(BATCH_SIZE, 3, 256, 256))
        labels = torch.zeros(BATCH_SIZE,1000).scatter(1,torch.randint(1,1000,(BATCH_SIZE,1)),1).to(device_id)
        loss_fn(outputs, labels).backward()
        optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()
