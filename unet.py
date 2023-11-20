import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        device = x.device
        # easy positional embedding like in Attention is All You Need
        half_dim = self.embed_dim // 2
        pos_code = torch.exp(
            torch.arange(half_dim, device=device) * (math.log(10000) / half_dim)
        )
        pos_code = torch.outer(x, pos_code)
        pos_code = torch.stack((pos_code[:, ::2].sin(), pos_code[:, 1::2].cos()), dim=-1)
        return pos_code


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        pass