import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, image_channels, betas, loss_type="l2"):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_steps = len(betas)
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod)
        denoise_coeff = betas / sqrt_one_minus_alphas_cumprod
        sigma = np.sqrt(betas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(sqrt_alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(sqrt_one_minus_alphas_cumprod))
        self.register_buffer("denoise_coeff", to_torch(denoise_coeff))
        self.register_buffer("sigma", to_torch(sigma))

    @torch.no_grad()
    def denoise(self, x, timesteps, y):
        return (x - self.denoise_coeff[timesteps] * self.model(x, timesteps, y)) / self.alphas[timesteps]

    @torch.no_grad()
    def sample(self, batch_size, device):
        x = torch.randn(batch_size, self.image_channels, self.image_size, self.image_size, device=device)
        for t in range(self.num_steps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.denoise(x, t_batch, None)
            noise = torch.randn_like(x)
            if t > 0:
                x = x + self.sigma[t] * noise
        return x

    @torch.no_grad()
    def sample_sequence(self, batch_size, device, y=None):
        x = torch.randn(batch_size, self.image_channels, self.image_size, self.image_size, device=device)
        sequence = [x]
        for t in range(self.num_steps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.denoise(x, t_batch, y)
            noise = torch.randn_like(x)
            if t > 0:
                x = x + self.sigma[t] * noise
            sequence.append(x)
        return sequence

    def get_losses(self, x, timesteps, y):
        noise = torch.randn_like(x)
        noised_x = x + self.sqrt_alphas_cumprod[timesteps] * x + self.sqrt_one_minus_alphas_cumprod[timesteps] * noise
        estimate_noise = self.denoise(noised_x, timesteps, y)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimate_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimate_noise, noise)
        else:
            raise NotImplementedError
        return loss

    def forward(self, x, y=None):
        bsz, ch, h, w = x.shape
        device = x.device
        timesteps = torch.randint(0, self.num_steps, (bsz,), device=device)
        loss = self.get_losses(x, timesteps, y)
        return loss
