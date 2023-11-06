import torch
from torch.utils.data import DataLoader

from dataloader.dataset import RandomDataset
from model.mlp_block import MLP

class Trainer():
    def __init__(self, 
                 task_type: str = 'imagenet',
                 batch_size: int = 32,
                 num_worker: int = 0,
                 shuffle: bool = True,
                 epoch: int = '3',
                 model_name: str = 'mlp',
                 learning_rate= 1e-4) -> None:
        
        self.BATCH_SIZE = batch_size
        self.EPOCH = epoch
        self.NUM_WORKER = num_worker
        self.SHUFFLE = shuffle
        self.LR = learning_rate
        
        self.model = self._build_model()
        self.dataloader = self._build_dataloader()
        self.opt = self._build_optimizer()
        self.loss_fn = self._build_loss()
    
    def _build_dataloader(self):
        dataset = RandomDataset(size=(5000,32,32,3))
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.BATCH_SIZE,
                                shuffle=self.SHUFFLE,
                                num_workers=self.NUM_WORKER)
        return dataloader
    
    def _build_model(self):
        model = MLP(in_channels=3072, hidden_channels=[128,64,32,10])
        return model
    
    def _build_optimizer(self):
        opt = torch.optim.Adam(self.model.parameters(), lr = self.LR)
        return opt
    
    def _build_loss(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn
    
    def train_step(self, x, label):
        x = torch.flatten(x, start_dim=1)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, label)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss
    
    def train(self):
        for e in range(self.EPOCH):
            for (i,data) in enumerate(self.dataloader):
                loss = self.train_step(data[0], data[1])
            print("Epoch {}, train loss: {}".format(e, loss))