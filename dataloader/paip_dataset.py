import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PAIPDataset(Dataset):
    def __init__(self, data_path, resolution):
        self.data_path = data_path
        self.resolution = resolution

        self.image_filenames = []
        self.mask_filenames = []

        for subdir in os.listdir(data_path):
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                image = os.path.join(subdir_path, f"rescaled_image_0_{resolution}x{resolution}.png")
                mask = os.path.join(subdir_path, f"rescaled_mask_0_{resolution}x{resolution}.png")

                # Ensure the image exist
                if os.path.exists(image) and os.path.exists(mask):

                    self.image_filenames.extend([image])
                    self.mask_filenames.extend([mask])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")  # Assuming masks are grayscale

        # Apply transformations
        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# Function to visualize a batch of images and masks
def visualize_samples(images, masks, num_samples=4):
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))
    for i in range(num_samples):
        image = F.to_pil_image(images[i])
        mask = F.to_pil_image(masks[i])

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

    plt.show()
    
if __name__ == "__main__":
    # Example usage
    data_path = "/Volumes/data/dataset/paip/output_images_and_masks"
    resolution = 512

    dataset = PAIPDataset(data_path, resolution)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in dataloader:
        images, masks = batch
        print(images.shape, masks.shape)
        # visualize_samples(images, masks, num_samples=4)
        # break
        # Your training/validation loop goes here