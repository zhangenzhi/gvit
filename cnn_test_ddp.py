import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
       
def train(gpu, args):
    # dist.init_process_group("nccl", init_method='env://', world_size=8 )
    # rank = dist.get_rank()
    # device_id = rank % torch.cuda.device_count()
    # model = ConvNet().to(device_id)
    # ddp_model = DDP(model, device_ids=[device_id])
    
    ############################################################
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    ############################################################
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    ###############################################################
    
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4*8)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./dataset',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    ################################################################
    
    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=0,
       pin_memory=True,
    #############################
      sampler=train_sampler)    # 
    #############################
    
    print(len(train_dataset))

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            if (i + 1) % 25 == 0 and gpu==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, GPU:{}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item(),
                    gpu),
                    )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        
def main():
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "23456"
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################
    # train(0, args)
    
if __name__ == '__main__':
    main()