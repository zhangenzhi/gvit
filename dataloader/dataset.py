import numpy as np
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(
        self, 
        size:  tuple = (5000, 32,32,3), #[num,w,h,c]
        ):
        
        self.size = size
        self.raw_x = np.random.rand(*size)
        self.raw_x = np.float32(self.raw_x)
        
        self.raw_y = np.random.randint(0, 9, size[0])
        self.raw_y = np.identity(10)[self.raw_y]
        self.raw_y = np.float32(self.raw_y)
    
    def __len__(self):
        return self.size[0]
    
    def __getitem__(self, index):
        return self.raw_x[index], self.raw_y[index]
    