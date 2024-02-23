import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from transunet import TransUNet
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from dataloader.paip_dataset import PAIPDataset

# Define the Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target) + self.smooth
        dice_coefficient = (2 * intersection + self.smooth) / union
        loss = 1.0 - dice_coefficient  # Adjusted to ensure non-negative loss
        return loss

# Create an instance of the U-Net model and other necessary components
input_channels = 3
num_classes = 1
unet_model = TransUNet(in_channels=input_channels, out_channels=num_classes)
criterion = DiceLoss()
optimizer = optim.Adam(unet_model.parameters(), lr=0.001)

# Split the dataset into train, validation, and test sets
data_path = "/Volumes/data/dataset/paip/output_images_and_masks"
resolution = 512

dataset = PAIPDataset(data_path, resolution)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = (dataset_size - train_size) // 2
test_size = dataset_size - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    unet_model.train()
    epoch_train_loss = 0.0

    for batch in train_loader:
        images, masks = batch
        optimizer.zero_grad()

        outputs = unet_model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation
    unet_model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            outputs = unet_model(images)
            loss = criterion(outputs, masks)
            epoch_val_loss += loss.item()

    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    # Additional diagnostics
    print("Sample Predictions:")
    unet_model.eval()
    with torch.no_grad():
        sample_images, sample_masks = next(iter(val_loader))
        sample_outputs = unet_model(sample_images)
        print("Sample Output Shape:", sample_outputs.shape)
        print("Sample Target Shape:", sample_masks.shape)

# Save train and validation losses
torch.save(train_losses, 'train_losses.pth')
torch.save(val_losses, 'val_losses.pth')

# Test the model
unet_model.eval()
test_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        images, masks = batch
        outputs = unet_model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
