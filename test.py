import torch
import torchvision
from torchvision import transforms


datapath = "/lustre/orion/gen006/proj-shared/enzhi/"

train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()  
    ])

imagenet_data = torchvision.datasets.ImageNet(datapath, transform= train_transform)
imagenet_val = torchvision.datasets.ImageNet(datapath, split="val", transform= train_transform)