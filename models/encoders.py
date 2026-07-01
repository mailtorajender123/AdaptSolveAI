import torch
import torch.nn as nn
from .adapter_layers import Adapter

class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 128, adapter_dim: int = 24, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(),
        )
        self.adapter = Adapter(latent_dim, adapter_dim, dropout)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.adapter(self.net(x))

class VisionEncoder(nn.Module):
    """Lightweight optional vision encoder for future CARLA/Habitat extension."""
    def __init__(self, in_channels=3, latent_dim=128, adapter_dim=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()
        )
        self.adapter = Adapter(latent_dim, adapter_dim)
    def forward(self, x):
        return self.adapter(self.conv(x))
