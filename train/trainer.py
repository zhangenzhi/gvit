import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader.dataset import RandomDataset
from model.mlp_block import MLP
from model.vision_transformer import VisionTransformer

class Trainer():
    def __init__(self, 
                 task_type: str = 'imagenet',
                 data_path: str = None,
                 batch_size: int = 32,
                 num_worker: int = 0,
                 shuffle: bool = True,
                 opt: str = "adam",
                 epoch: int = 10,
                 model_name: str = 'mlp',
                 learning_rate= 1e-4) -> None:
        
        self.TASK = task_type
        self.DATA_PATH = data_path
        self.BATCH_SIZE = batch_size
        self.EPOCH = epoch
        self.NUM_WORKER = num_worker
        self.SHUFFLE = shuffle
        self.OPT = opt
        self.LR = learning_rate
        self.MODEL = model_name
        
        self.device = self._check_device_avail()
        self.model = self._build_model()
        self.train_dataloader, self.val_dataloader = self._build_dataloader()
        self.opt = self._build_optimizer()
        self.loss_fn = self._build_loss()
    
    def _check_device_avail(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Devices found {torch.cuda.device_count()}.")
        else:
            device = torch.device('cpu')
            print("No device found, use cpu only")
        return device
            
    def _build_dataloader(self):
        if self.TASK == "random":
            self.NUM_CLASS = 10
            train_dataset = RandomDataset(size=(5000,3,32,32), num_class=10)
            val_dataset = RandomDataset(size=(1000,3,32,32), num_class=10)
        elif self.TASK == "fake_imagenet":
            self.NUM_CLASS = 1000
            train_dataset = RandomDataset(size=(5000,3,256,256), num_class=1000)
            val_dataset = RandomDataset(size=(1000,3,256,256), num_class=1000)
        elif self.TASK == "imagenet":
            self.NUM_CLASS = 1000
            train_transform = transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.ToTensor()  
                    ])
            train_dataset = torchvision.datasets.ImageNet(self.DATA_PATH, transform= train_transform)
            val_dataset = torchvision.datasets.ImageNet(self.DATA_PATH, split="val", transform= train_transform)
            
        train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=self.BATCH_SIZE,
                                shuffle=self.SHUFFLE,
                                num_workers=self.NUM_WORKER)
        
        val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=self.BATCH_SIZE,
                                shuffle=self.SHUFFLE,
                                num_workers=self.NUM_WORKER)
        return train_dataloader, val_dataloader
    
    def _build_model(self):
        if self.MODEL == 'mlp':
            model = MLP(in_channels=3072, hidden_channels=[128,64,32,10])
        elif self.MODEL == "vit":
            model = VisionTransformer(
                            image_size=256,
                            patch_size=32,
                            num_layers=12,
                            num_heads=12,
                            hidden_dim=3072,
                            mlp_dim=3072,
                            num_classes=1000)
        return model
    
    def _build_optimizer(self):
        opt = torch.optim.Adam(self.model.parameters(), lr = self.LR)
        return opt
    
    def _build_loss(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn
    
    
    def train(self):
        
        self.model.to(self.device)
        
        for e in range(self.EPOCH):
            correct = 0
            for (i,data) in enumerate(self.train_dataloader):
                x,label = data[0], data[1]
                label = torch.reshape(label,(self.BATCH_SIZE,1))
                label = torch.zeros(self.BATCH_SIZE,self.NUM_CLASS).scatter(1,label,1)
                x, label = x.cuda(), label.cuda()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, label)
                loss.backward()
                self.opt.step()
                correct += (y_pred == label).float().sum()
                self.opt.zero_grad()
                if i%100==0:
                    print(f"Step {i}, train loss: {loss}.")
            accuracy = 100 * correct / len(self.train_dataloader)
            print("Epoch {}, train loss: {}, train acc: {}".format(e, loss, accuracy))
            print(f"Epoch {e}, train loss: {loss}, train acc: {accuracy}")