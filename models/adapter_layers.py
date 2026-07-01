import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int = 24, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.dropout(self.act(self.down(x))))


def freeze_except_adapters_bn(model: nn.Module):
    for _, p in model.named_parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, (Adapter, nn.BatchNorm1d, nn.LayerNorm)):
            for p in module.parameters(recurse=True):
                p.requires_grad = True
    # Keep small policy adapter trainable, not full policy.
    return [p for p in model.parameters() if p.requires_grad]


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True
