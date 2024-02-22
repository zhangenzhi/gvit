import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from dataloader.paip_dataset import PAIPDataset


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input)  # 64,256,256
        e2 = self.layer2(e1)     # 64,128,128
        e3 = self.layer3(e2)     # 128,64,64
        e4 = self.layer4(e3)     # 256,32,32
        f = self.layer5(e4)      # 512,16,16
        d4 = self.decode4(f, e4) # 256,32,32
        d3 = self.decode3(d4, e3) # 256,64,64
        d2 = self.decode2(d3, e2) # 128,128,128
        d1 = self.decode1(d2, e1) # 64,256,256
        d0 = self.decode0(d1)     # 64,512,512
        out = self.conv_last(d0)  # 1,512,512
        return out

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
unet_model = Unet(n_class=1)
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
