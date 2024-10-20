import torch
import torch.nn as nn
import torch.nn.functional as F

class AntiRectifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)  # Subtract mean along dimension 1
        x = F.normalize(x, p=2, dim=1)  # Normalize across dimension 1
        pos_neg = torch.cat([F.relu(x), F.relu(-x)], dim=1)  # Combine positive and negative rectified parts
        return pos_neg

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = AntiRectifier()

        self.conv2 = nn.Conv2d(2 * mid_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.act2 = AntiRectifier()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim)
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x_res + self.dropout1(x)

        x_res = x
        x = self.norm2(x)
        x = x + self.dropout2(self.mlp(x))
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class TransformerCNN(nn.Module):
    def __init__(self, n_channels=128, embed_dim=512, num_heads=8, mlp_dim=1024, num_transformer_layers=4, bilinear=False):
        super(TransformerCNN, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Transformer encoder
        self.flatten = nn.Flatten(2)
        self.embed_dim = embed_dim
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_transformer_layers)]
        )

        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        # CNN Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Flatten and add positional encoding for Transformer
        b, c, h, w = x4.shape
        seq_len = h * w
        x_flat = x4.flatten(2).permute(2, 0, 1)  # Shape: (seq_len, batch_size, channels)

        # Create positional embedding dynamically
        pos_embedding = nn.Parameter(torch.zeros(seq_len, 1, self.embed_dim)).to(x.device)
        x_flat = x_flat + pos_embedding

        # Transformer Encoder
        for transformer in self.transformer_blocks:
            x_flat = transformer(x_flat)

        x4 = x_flat.permute(1, 2, 0).view(b, c, h, w)

        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return F.softmax(x, dim=1)
