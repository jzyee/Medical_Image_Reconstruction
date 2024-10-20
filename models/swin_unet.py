import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Define the basic building block of Swin Transformer
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Apply multi-head self-attention
        x = x.view(B, self.num_heads, L // self.num_heads, C)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output.view(B, L, C)

        x = shortcut + x  # Residual connection
        x = x + self.mlp(self.norm2(x))  # MLP layer
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_chans=128, embed_dim=96, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class SwinUNet(nn.Module):
    def __init__(self, in_chans=128, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], patch_size=4):
        super(SwinUNet, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.layers = nn.ModuleList()

        # Build layers based on depth and number of heads
        for i, depth in enumerate(depths):
            layer = nn.ModuleList([SwinBlock(embed_dim * (2 ** i), num_heads[i]) for _ in range(depth)])
            self.layers.append(layer)

        self.final_layer = nn.Conv2d(embed_dim * (2 ** (len(depths) - 1)), in_chans, kernel_size=1)

    def preprocess_input(self, x):
        # x shape: (B, HW, C)
        B, HW, C = x.shape
        H = W = int(HW ** 0.5)
        
        # Reshape to (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        # If the spatial dimensions are not divisible by patch_size, pad the input
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        return x

    def forward(self, x):
        # Preprocess the input
        x = self.preprocess_input(x)
        
        # Apply patch embedding
        x, (H, W) = self.patch_embed(x)

        # Apply Swin Transformer blocks
        for layer in self.layers:
            for block in layer:
                x = block(x)

        # Reshape and apply final convolution
        x = x.permute(0, 2, 1).view(-1, self.embed_dim * (2 ** (len(self.layers) - 1)), H, W)
        x = self.final_layer(x)
        
        # Ensure output has the same spatial dimensions as input
        if x.shape[2:] != (H * self.patch_size, W * self.patch_size):
            x = F.interpolate(x, size=(H * self.patch_size, W * self.patch_size), mode='bilinear', align_corners=False)
        
        # Reshape output to match input shape (B, HW, C)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        
        return x
