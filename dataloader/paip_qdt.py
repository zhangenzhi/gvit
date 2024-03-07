import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PAIQDTDataset(Dataset):
    def __init__(self, data_path, resolution, normalize=False):
        self.data_path = data_path
        self.resolution = resolution

        self.image_filenames = []
        self.qdt_filenames = []
        self.mask_filenames = []

        for subdir in os.listdir(data_path):
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                image = os.path.join(subdir_path, f"rescaled_image_0_{resolution}x{resolution}.png")
                qdt =  os.path.join(subdir_path, f"{resolution}_qdt.png")
                mask = os.path.join(subdir_path, f"rescaled_mask_0_{resolution}x{resolution}.png")

                # Ensure the image exist
                if os.path.exists(image) and os.path.exists(mask):

                    self.image_filenames.extend([image])
                    self.qdt_filenames.extend([qdt])
                    self.mask_filenames.extend([mask])

        # Compute mean and std from the dataset (you need to implement this)
        self.mean, self.std = self.compute_img_statistics()
        self.mask_m, self.mask_std = self.compute_mask_statistics()

        self.transform= transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.transform_mask= transforms.Compose([
            transforms.ToTensor(),
        ])

        if normalize:
            self.transform.transforms.append(transforms.Normalize(mean=self.mean, std=self.std))
            # self.transform_mask.transforms.append(transforms.Normalize(mean=self.mask_m, std=self.mask_std))
    
    
    def compute_img_statistics(self):
        # Initialize accumulators for mean and std
        mean_acc = np.zeros(3)
        std_acc = np.zeros(3)

        # Loop through the dataset and accumulate channel-wise mean and std
        for img_name in self.image_filenames:
            img = Image.open(img_name).convert("RGB")
            img_np = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            
            # Accumulate mean and std separately for each channel
            mean_acc += np.mean(img_np, axis=(0, 1))
            std_acc += np.std(img_np, axis=(0, 1))

        # Calculate the overall mean and std
        mean = mean_acc / len(self.image_filenames)
        std = std_acc / len(self.image_filenames)

        return mean.tolist(), std.tolist()
    
    def compute_mask_statistics(self):
        # Initialize accumulators for mean and std
        mean_acc = np.zeros(1)
        std_acc = np.zeros(1)

        # Loop through the dataset and accumulate channel-wise mean and std
        for img_name in self.mask_filenames:
            img = Image.open(img_name).convert("L")
            img_np = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            
            # Accumulate mean and std separately for each channel
            mean_acc += np.mean(img_np, axis=(0, 1))
            std_acc += np.std(img_np, axis=(0, 1))

        # Calculate the overall mean and std
        mean = mean_acc / len(self.mask_filenames)
        std = std_acc / len(self.mask_filenames)

        return mean.tolist(), std.tolist()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        qdt_name = self.qdt_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image = Image.open(img_name).convert("RGB")
        qdt =  Image.open(qdt_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")  # Assuming masks are grayscale

        # Apply transformations
        image = self.transform(image)
        qdt = self.transform(qdt)
        mask = self.transform_mask(mask)

        return image, qdt, mask

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
    
if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="paip", 
                        help='base path of dataset.')
    parser.add_argument('--datapath', default="/Volumes/data/dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=512, type=int,
                        help='resolution of img.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./vitunet_visual",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    dataset = PAIQDTDataset(args.datapath, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in dataloader:
        images, qdt, masks = batch
        print(images.shape, qdt.shape, masks.shape)
        # visualize_samples(images, masks, num_samples=4)
        # break
        # Your training/validation loop goes here