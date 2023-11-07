import torch
from torch.utils.data import DataLoader

from dataloader.dataset import RandomDataset
from model.mlp_block import MLP
from model.vision_transformer import VisionTransformer

class Trainer():
    def __init__(self, 
                 task_type: str = 'imagenet',
                 batch_size: int = 32,
                 num_worker: int = 0,
                 shuffle: bool = True,
                 epoch: int = '3',
                 model_name: str = 'mlp',
                 learning_rate= 1e-4) -> None:
        
        self.TASK = task_type
        self.BATCH_SIZE = batch_size
        self.EPOCH = epoch
        self.NUM_WORKER = num_worker
        self.SHUFFLE = shuffle
        self.LR = learning_rate
        self.MODEL = model_name
        
        self.model = self._build_model()
        self.dataloader = self._build_dataloader()
        self.opt = self._build_optimizer()
        self.loss_fn = self._build_loss()
    
    def _build_dataloader(self):
        if self.TASK == "random":
            dataset = RandomDataset(size=(5000,32,32,3),num_class=10)
        elif self.TASK == "imagenet":
            dataset = RandomDataset(size=(5000,3,256,256), num_class=1000)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.BATCH_SIZE,
                                shuffle=self.SHUFFLE,
                                num_workers=self.NUM_WORKER)
        return dataloader
    
    def _build_model(self):
        if self.MODEL == 'mlp':
            model = MLP(in_channels=3072, hidden_channels=[128,64,32,10])
        elif self.MODEL == "vit":
            model = VisionTransformer(
                            image_size=256,
                            patch_size=32,
                            num_layers=3,
                            num_heads=4,
                            hidden_dim=128,
                            mlp_dim=128,
                            num_classes=1000)
        return model
    
    def _build_optimizer(self):
        opt = torch.optim.Adam(self.model.parameters(), lr = self.LR)
        return opt
    
    def _build_loss(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn
    
    # def train_step(self, x, label, correct):
    #     # x = torch.flatten(x, start_dim=1)
    #     y_pred = self.model(x)
    #     loss = self.loss_fn(y_pred, label)
    #     loss.backward()
    #     self.opt.step()
    #     correct += (y_pred == label).float().sum()
    #     self.opt.zero_grad()
    #     return loss, correct
    
    def train(self):
        for e in range(self.EPOCH):
            correct = 0
            for (i,data) in enumerate(self.dataloader):
                x,label = data[0], data[1]
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, label)
                loss.backward()
                self.opt.step()
                correct += (y_pred == label).float().sum()
                self.opt.zero_grad()
            accuracy = 100 * correct / len(self.dataloader)
            print("Epoch {}, train loss: {}, train acc: {}".format(e, loss, accuracy))
            print(f"Epoch {e}, train loss: {loss}, train acc: {accuracy}")