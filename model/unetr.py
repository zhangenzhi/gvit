import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
sys.path.append("./")
import math
import torch.nn.functional as F

from model.vit import TransformerEncoderBlock, TransformerEncoder


class EncoderBottleneck(nn.Module):
    def __init__(self,  embedding_dim, head_num, mlp_dim, num_blocks=3, proj=True):
        super().__init__()
        self.proj=proj
        self.encoder_block = nn.ModuleList([TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(num_blocks)])
        if proj:
            self.projection =  nn.Linear(embedding_dim, embedding_dim*2)
            
    def forward(self, x):
        for blk in self.encoder_block:
            x = blk(x)
        if self.proj:
            x = self.projection(x)
        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, skips=1):
        super().__init__()

        if scale_factor==2:
            self.upsample = nn.ConvTranspose2d(in_channels=in_channels//2, out_channels=in_channels//2, 
                                            kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, 
                                kernel_size=4, stride=2, padding=1)
        self.upsample_skips = nn.ModuleList([nn.ConvTranspose2d(in_channels=in_channels//2, out_channels=in_channels//2, 
                                                                kernel_size=4, stride=2, padding=1) for _ in range(skips)])
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        
        if x_concat is not None:
            for skp in self.upsample_skips:  
                x_concat = skp(x_concat)
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_size,
                 classification=True, num_classes=1):
        super().__init__()

        self.patch_size = patch_size
        self.classification = classification
        
        # patches conv
        self.conv1 = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm1 = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # temp
        self.token_dim = img_dim//patch_size * img_dim//patch_size
        self.vit_img_dim = int(math.sqrt(self.token_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim*4))
        
        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.vit_img_dim*self.vit_img_dim + 1, embedding_dim*4))
        
        self.encoder1 = EncoderBottleneck(embedding_dim, head_num, mlp_dim)
        self.encoder2 = EncoderBottleneck(embedding_dim*2, head_num, mlp_dim)
        self.encoder3 = EncoderBottleneck(embedding_dim*4, head_num, mlp_dim, proj=False)

        self.img_proj = nn.Linear(self.token_dim, self.vit_img_dim*self.vit_img_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(embedding_dim*4, head_num, mlp_dim, block_num)
        
        self.norm2 = nn.BatchNorm2d(self.vit_img_dim*self.vit_img_dim)
        
        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x=[8,3,8,4096]
        x = self.conv1(x) # x=[8,1024,1,512]
        x = self.norm1(x)
    
        batch_size, channels, h, w = x.shape # [8,1024,1,529]
        tokens = h*w
        x= x.permute(0, 3, 1, 2).contiguous().view(batch_size, tokens, channels) # [b,tokens,emb]  [8,529,1024]
        
        x1 = self.relu(x)  # [b,tokens,emb]  [8,529,128]
        x2 = self.encoder1(x1)  # [b,tokens,emb]  [8,529,258]
        x3 = self.encoder2(x2)  # [b,tokens,emb]  [8,529,512]
        x = self.encoder3(x3)  # [b,tokens,emb]  [8,529,512]
        # x, x1, x2, x3 : (batch_size, tokens, emb)
        
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)  # [2, 1, 128]
        patches = torch.cat([token, x], dim=1)  # [b, tokens+1, emb]  [8,530,128]
        patches += self.embedding[:tokens + 1, :]
        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]
        
        x  = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim) # [2, 128, 23, 23]
        x1 = rearrange(x1, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim) # [2, 128, 23, 23]
        x2 = rearrange(x2, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim) # [2, 128, 23, 23]
        x3 = rearrange(x3, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim) # [2, 128, 23, 23]

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2, skips=1)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels, skips=2)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2), skips=3)
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8), scale_factor=1)

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class UNetr(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_size, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_size, classification=False, num_classes=1)

        self.decoder = Decoder(out_channels, class_num)
        self.img_dim = img_dim
        self.upsampling = nn.Upsample(size=(self.img_dim, self.img_dim), mode='bilinear', align_corners=True)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)
        x = self.upsampling(x)
     
        return x


if __name__ == '__main__':
    import torch

    unetr = UNetr(img_dim=512,
                in_channels=3,
                out_channels=128,
                head_num=4,
                mlp_dim=512,
                block_num=3,
                patch_size=4,
                class_num=1)

    print(sum(p.numel() for p in unetr.parameters()))
    print(unetr(torch.randn(1, 3, 512, 512)).shape)
    
    from calflops import calculate_flops
    batch_size = 1
    input_shape = (batch_size, 3, 512, 512)
    flops, macs, params = calculate_flops(model=unetr, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("Unetr FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))