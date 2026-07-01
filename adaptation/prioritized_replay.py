from __future__ import annotations
import numpy as np
import torch

class ShiftAwarePrioritizedReplay:
    def __init__(self, capacity=10000, alpha=0.6, novelty_w=0.5, safety_w=0.3, fisher_w=0.2, seed=42):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.novelty_w = float(novelty_w)
        self.safety_w = float(safety_w)
        self.fisher_w = float(fisher_w)
        self.rng = np.random.default_rng(seed)
        self.data = []
        self.priorities = []

    def __len__(self):
        return len(self.data)

    def add(self, state, action, reward, next_state, done, safety_margin=1.0, novelty=1.0, fisher=0.0):
        safety_prox = 1.0 / (abs(float(safety_margin)) + 1e-3)
        priority = self.novelty_w * float(novelty) + self.safety_w * safety_prox + self.fisher_w * float(fisher) + 1e-6
        item = {
            'state': np.asarray(state, dtype=np.float32),
            'action': int(action),
            'reward': float(reward),
            'next_state': np.asarray(next_state, dtype=np.float32),
            'done': float(done),
            'safety_margin': float(safety_margin),
            'priority': float(priority),
        }
        if len(self.data) >= self.capacity:
            self.data.pop(0); self.priorities.pop(0)
        self.data.append(item); self.priorities.append(float(priority))

    def sample(self, batch_size, device='cpu'):
        if not self.data:
            raise ValueError('Replay buffer is empty')
        p = np.asarray(self.priorities, dtype=np.float64)
        p = np.power(np.maximum(p, 1e-8), self.alpha)
        p = p / p.sum()
        idx = self.rng.choice(len(self.data), size=min(batch_size, len(self.data)), replace=len(self.data) < batch_size, p=p)
        batch = [self.data[i] for i in idx]
        return {
            'state': torch.tensor(np.stack([b['state'] for b in batch]), dtype=torch.float32, device=device),
            'action': torch.tensor([b['action'] for b in batch], dtype=torch.long, device=device),
            'reward': torch.tensor([b['reward'] for b in batch], dtype=torch.float32, device=device),
            'next_state': torch.tensor(np.stack([b['next_state'] for b in batch]), dtype=torch.float32, device=device),
            'done': torch.tensor([b['done'] for b in batch], dtype=torch.float32, device=device),
            'safety_margin': torch.tensor([b['safety_margin'] for b in batch], dtype=torch.float32, device=device),
        }
