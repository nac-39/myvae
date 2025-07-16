import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Params:
    input_dim: int = 784
    hidden_dim: int = 200
    latent_dim: int = 20
    epochs: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 32


class Encoder(nn.Module):
    def __init__(self, p: Params):
        super().__init__()
        self.linear = nn.Linear(p.input_dim, p.hidden_dim)
        self.linear_mu = nn.Linear(p.hidden_dim, p.latent_dim)
        self.linear_logvar = nn.Linear(p.hidden_dim, p.latent_dim)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)  # 分散が正であるという制約を満たすため
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, p: Params):
        super().__init__()
        self.linear1 = nn.Linear(p.latent_dim, p.hidden_dim)
        self.linear2 = nn.Linear(p.hidden_dim, p.input_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        x = self.linear2(h)
        x_hat = F.sigmoid(x)
        return x_hat


class VAE(nn.Module):
    def __init__(self, p: Params):
        super().__init__()
        self.encoder = Encoder(p)
        self.decoder = Decoder(p)

    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        return (L1 + L2) / batch_size

    @staticmethod
    def reparameterize(mu, sigma):
        """変数変換トリック"""
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
