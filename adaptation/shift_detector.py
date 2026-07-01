from __future__ import annotations
import numpy as np
import torch

class ShiftEntropyDetector:
    def __init__(self, latent_dim: int, shift_threshold=3.0, entropy_threshold=None, eps=1e-5):
        self.latent_dim = latent_dim
        self.shift_threshold = float(shift_threshold)
        self.entropy_threshold = entropy_threshold
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(latent_dim, dtype=np.float64)
        self.var = np.ones(latent_dim, dtype=np.float64)
        self.entropies = []

    def update_stats(self, z, entropy=None):
        z_np = z.detach().cpu().numpy().reshape(-1, self.latent_dim)
        for row in z_np:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.var += (delta * delta2 - self.var) / max(self.count, 1)
        if entropy is not None:
            self.entropies.append(float(entropy))
            if len(self.entropies) > 2000:
                self.entropies = self.entropies[-2000:]

    def calibrate_entropy(self, quantile=0.90):
        if self.entropies:
            self.entropy_threshold = float(np.quantile(self.entropies, quantile))
        return self.entropy_threshold

    def score(self, z):
        row = z.detach().cpu().numpy().reshape(-1, self.latent_dim).mean(axis=0)
        dist = np.sqrt(np.mean(((row - self.mean) ** 2) / (self.var + self.eps)))
        return float(dist)

    def should_adapt(self, z, entropy):
        shift_score = self.score(z)
        entropy_val = float(entropy.detach().cpu().numpy().mean() if torch.is_tensor(entropy) else entropy)
        ent_thr = self.entropy_threshold if self.entropy_threshold is not None else np.inf
        trigger = (shift_score > self.shift_threshold) or (entropy_val > ent_thr)
        return bool(trigger), {'shift_score': shift_score, 'entropy': entropy_val, 'entropy_threshold': ent_thr}
