import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
sys.path.append("./")
import math
import torch.nn.functional as F

from model.vit import TransformerEncoderBlock, TransformerEncoder
from model.seg_mask import MaskTransformer

class EncoderBottleneck(nn.Module):
    def __init__(self,  embedding_dim, head_num, mlp_dim):
        super().__init__()
        self.encoder_block = TransformerEncoderBlock(embedding_dim, head_num, mlp_dim)
        
    def forward(self, x):
        x = self.encoder_block(x)
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
        self.token_dim = 512
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(img_dim + 1, embedding_dim))
        
        self.encoder1 = EncoderBottleneck(embedding_dim, head_num, mlp_dim)
        self.encoder2 = EncoderBottleneck(embedding_dim, head_num, mlp_dim)
        self.encoder3 = EncoderBottleneck(embedding_dim, head_num, mlp_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)
        
        self.norm2 = nn.BatchNorm2d(self.token_dim)
        
        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x=[8,3,8,4096]
        x = self.conv1(x) # x=[8, 1024, 1, 512]
        x = self.norm1(x)
        batch_size, channels, _, tokens = x.shape # [8,1024,1,529]
        x= x.permute(0, 3, 1, 2).contiguous().view(batch_size, tokens, channels) # [b,tokens,emb]  [8,529,1024]
        
        x1 = self.relu(x)  # [b,tokens,emb]  [8,529,128]
        x2 = self.encoder1(x1)  # [b,tokens,emb]  [8,529,512]
        x3 = self.encoder2(x2)  # [b,tokens,emb]  [8,529,512]
        x4 = self.encoder3(x3)  # [b,tokens,emb]  [8,529,512]
        # x, x1, x2, x3 : (batch_size, tokens, emb)
        
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)  # [2, 1, 128]
        patches = torch.cat([token, x4], dim=1)  # [b, tokens+1, emb]  [8,530,128]
        patches += self.embedding[:tokens + 1, :]
        x4 = self.dropout(patches)
        x4 = self.transformer(x4)
        x4 = self.mlp_head(x4[:, 0, :]) if self.classification else x4[:, 1:, :]
        # x4 = rearrange(x4, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim) # [2, 128, 23, 23]

        return x4


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.masked_decoder = MaskTransformer(n_cls=1, patch_size=32, d_encoder=512, n_heads=8,
                                              n_layers=8, d_model=512, d_ff=4*512, drop_path_rate=0.0, 
                                              dropout=0.0)

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x):
        x = self.masked_decoder(x, im_size=(512,512))
        return x


class ViTSeg(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_size, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_size, classification=False, num_classes=1)

        self.decoder = Decoder(out_channels, class_num)
        self.img_dim = img_dim
        self.upsampling = nn.Upsample(size=(self.img_dim, self.img_dim), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsampling(x)

        return x


if __name__ == '__main__':
    import torch

    vitunet = ViTSeg(img_dim=512,
                        in_channels=3,
                        out_channels=128,
                        head_num=4,
                        mlp_dim=512,
                        block_num=8,
                        patch_size=16,
                        class_num=1)

    print(sum(p.numel() for p in vitunet.parameters()))
    print(vitunet(torch.randn(1, 3, 512, 512)).shape)