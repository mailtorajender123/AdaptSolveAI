import torch
import torch.nn as nn
from .adapter_layers import Adapter

class PolicyHead(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, num_actions: int = 4, adapter_dim: int = 24, dropout: float = 0.05):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            Adapter(hidden_dim, adapter_dim, dropout),
        )
        self.action = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = self.body(z)
        return self.action(h), self.value(h).squeeze(-1)
