import torch
from train.trainer import Trainer

class ViTTrainer(Trainer):
    def __init__(
        self, 
        task_type: str = 'imagenet', 
        data_path: str = None,
        batch_size: int = 32,
        num_worker: int = 8, 
        shuffle: bool = True, 
        epoch: int = '100', 
        opt = 'adam',
        model_name: str = 'vit', 
        learning_rate=1e-4) -> None:
        super().__init__(task_type, batch_size, num_worker, shuffle, opt, epoch, model_name, learning_rate)
    
    def _build_dataloader(self):
        return super()._build_dataloader()
    
    
    
    
    