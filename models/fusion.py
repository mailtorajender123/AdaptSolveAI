import torch
import torch.nn as nn
from .adapter_layers import Adapter

class MultiModalAttentionFusion(nn.Module):
    def __init__(self, latent_dim: int = 128, num_heads: int = 4, adapter_dim: int = 24, dropout: float = 0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.adapter = Adapter(latent_dim, adapter_dim, dropout)
        self.gate = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Sigmoid())

    def forward(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            x = embeddings.unsqueeze(1)
        else:
            x = torch.stack(embeddings, dim=1)
        attn_out, weights = self.attn(x, x, x, need_weights=True)
        pooled = attn_out.mean(dim=1)
        pooled = self.norm(pooled)
        pooled = self.adapter(pooled)
        return pooled * self.gate(pooled), weights
