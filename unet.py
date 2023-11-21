from turtle import forward
from sklearn import base
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
    def __init__(self, in_channels, num_groups):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, c)
        k = k.reshape(b, c, h * w)
        v = v.permute(0, 2, 3, 1).reshape(b, h * w, c)

        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)
        out = torch.bmm(attn, v).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return self.to_out(out) + x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, timestep_embed_dim=None, num_classes=None, num_groups=32, use_attention=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_bias = nn.Embedding(timestep_embed_dim, out_channels) if timestep_embed_dim is not None else None
        self.class_bias = nn.Linear(num_classes, out_channels) if num_classes is not None else None
        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.attention = AttentionBlock(out_channels, num_groups) if use_attention else nn.Identity()

    def forward(self, x, timestep_embed=None, y=None):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.time_bias is not None:
            out = out + self.time_bias(timestep_embed).reshape(1, -1, 1, 1)
        if self.class_bias is not None:
            out = out + self.class_bias(y).reshape(-1, -1, 1, 1)
        out = out + self.residual_connection(x)
        return self.attention(out)
    

class UNet(nn.Module):
    def __init__(self, image_channels, base_channels, channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2, timestep_dim=None, num_classes=None, dropout=0.1, 
                 attention_resolutions=(), num_groups=32, pad=0):
        super().__init__()
        self.pad = pad
        self.num_classes = num_classes
        self.time_embed = nn.Sequential(
            PositionalEmbedding(base_channels),
            nn.Linear(base_channels, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, timestep_dim)
        ) if timestep_dim is not None else None

        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels, out_channels, dropout,
                    timestep_embed_dim=timestep_dim,
                    num_classes=num_classes,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions
                ))
                now_channels = out_channels
                channels.append(out_channels)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                now_channels = out_channels

        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=timestep_dim,
                num_classes=num_classes,
                num_groups=num_groups,
                use_attention=True,
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=timestep_dim,
                num_classes=num_classes,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=timestep_dim,
                    num_classes=num_classes,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            
            if i != 0:
                self.ups.append(Upsample(now_channels))

        self.outnorm = nn.GroupNorm(num_groups, base_channels)
        self.outconv = nn.Conv2d(base_channels, image_channels, 3, padding=1)

    def forward(self, x, timestep=None, y=None):
        if self.pad != 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        if self.time_embed is not None:
            timestep_embed = self.time_embed(timestep)
        else:
            timestep_embed = None
        x = self.init_conv(x)
        skips = [x]
        for down in self.downs:
            x = down(x, timestep_embed, y)
            skips.append(x)
        for mid in self.mid:
            x = mid(x, timestep_embed, y)
        for up in self.ups:
            x = up(torch.cat([x, skips.pop()], dim=1), timestep_embed, y)
        x = self.outnorm(x)
        x = F.relu(x)
        return self.outconv(x)[:, :, self.pad:-self.pad, self.pad:-self.pad] if self.pad != 0 else self.outconv(x)
