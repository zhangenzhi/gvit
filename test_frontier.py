import torch
import torchvision
from torchvision import transforms


print(torch.cuda.get_device_name())

datapath = "/lustre/orion/gen006/proj-shared/enzhi/"

train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()  
    ])

imagenet_data = torchvision.datasets.ImageNet(datapath, transform= train_transform)
imagenet_val = torchvision.datasets.ImageNet(datapath, split="val", transform= train_transform)

print("train samples:{}, val_samples:{}".format(len(imagenet_data), len(imagenet_val)))


train_loader = torch.utils.data.DataLoader(imagenet_data,
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers=64)
val_loader = torch.utils.data.DataLoader(imagenet_val,
                                    batch_size=4,
                                    shuffle=True,
                                    num_workers=64)
train_dataset = iter(train_loader)
val_dataset = iter(val_loader)