import torch
import torch.nn as nn
import torch.nn.functional as F

class CPCHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
    def forward(self, z_t, z_future, temperature=0.2):
        pred = F.normalize(self.predictor(z_t), dim=-1)
        target = F.normalize(z_future.detach(), dim=-1)
        logits = pred @ target.t() / temperature
        labels = torch.arange(z_t.size(0), device=z_t.device)
        return F.cross_entropy(logits, labels)

class MAEHead(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, input_dim))
    def forward(self, z, target_x, mask):
        recon = self.decoder(z)
        denom = mask.sum().clamp_min(1.0)
        return (torch.abs((recon - target_x) * mask).sum() / denom)

class DistillationHead(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, student_logits, teacher_logits, temperature=2.0):
        s = F.log_softmax(student_logits / temperature, dim=-1)
        t = F.softmax(teacher_logits.detach() / temperature, dim=-1)
        return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)

class SafetyAwareHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.margin_predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim//2), nn.GELU(), nn.Linear(latent_dim//2, 1))
    def forward(self, z, safety_margin):
        pred = self.margin_predictor(z).squeeze(-1)
        regression = F.smooth_l1_loss(pred, safety_margin.detach())
        violation_penalty = F.relu(-pred).mean()
        return regression + violation_penalty

class CrossModalAlignmentHead(nn.Module):
    def forward(self, z_a, z_b, temperature=0.2):
        za = F.normalize(z_a, dim=-1)
        zb = F.normalize(z_b, dim=-1)
        logits = za @ zb.t() / temperature
        labels = torch.arange(z_a.size(0), device=z_a.device)
        return F.cross_entropy(logits, labels)
