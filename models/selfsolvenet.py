from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import StateEncoder
from .fusion import MultiModalAttentionFusion
from .policy_head import PolicyHead
from .ssl_heads import CPCHead, MAEHead, DistillationHead, SafetyAwareHead, CrossModalAlignmentHead

class SelfSolveNet(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int = 128, hidden_dim: int = 128,
                 num_actions: int = 4, num_heads: int = 4, adapter_dim: int = 24, dropout: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.state_encoder = StateEncoder(state_dim, latent_dim, hidden_dim, adapter_dim, dropout)
        self.fusion = MultiModalAttentionFusion(latent_dim, num_heads, adapter_dim, dropout)
        self.policy = PolicyHead(latent_dim, hidden_dim, num_actions, adapter_dim, dropout)
        self.cpc = CPCHead(latent_dim)
        self.mae = MAEHead(state_dim, latent_dim)
        self.distill = DistillationHead()
        self.safety_head = SafetyAwareHead(latent_dim)
        self.xmodal = CrossModalAlignmentHead()

    def encode(self, state):
        e = self.state_encoder(state)
        z, attn = self.fusion(e)
        return z, attn

    def forward(self, state):
        z, attn = self.encode(state)
        logits, value = self.policy(z)
        return {'z': z, 'logits': logits, 'value': value, 'attn': attn}

    def act(self, state, deterministic=False, feasible_mask=None):
        out = self.forward(state)
        logits = out['logits']
        if feasible_mask is not None:
            mask = torch.as_tensor(feasible_mask, dtype=torch.bool, device=logits.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.distributions.Categorical(probs).sample()
        log_prob = torch.log(probs.gather(1, action.view(-1, 1)).clamp_min(1e-8)).squeeze(-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(-1)
        return action, log_prob, entropy, out

    def compute_ssl_loss(self, batch, cfg, teacher=None):
        states = batch['state']
        next_states = batch.get('next_state', states)
        safety_margin = batch.get('safety_margin', torch.ones(states.size(0), device=states.device))
        out = self.forward(states)
        next_out = self.forward(next_states)
        z = out['z']
        mask_ratio = float(cfg.get('mask_ratio', 0.25))
        mask = (torch.rand_like(states) < mask_ratio).float()
        masked_states = states * (1.0 - mask)
        z_masked = self.forward(masked_states)['z']
        cpc_loss = self.cpc(z, next_out['z'], cfg.get('temperature', 0.2))
        mae_loss = self.mae(z_masked, states, mask)
        safety_loss = self.safety_head(z, safety_margin)
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(states)['logits']
            distill_loss = self.distill(out['logits'], teacher_logits)
        else:
            distill_loss = torch.zeros((), device=states.device)
        total = (cfg.get('cpc_weight', 1.0) * cpc_loss + cfg.get('mae_weight', 0.5) * mae_loss +
                 cfg.get('distill_weight', 0.3) * distill_loss + cfg.get('safety_weight', 1.0) * safety_loss)
        return total, {'cpc': float(cpc_loss.detach()), 'mae': float(mae_loss.detach()),
                       'distill': float(distill_loss.detach()), 'safety': float(safety_loss.detach()),
                       'ssl_total': float(total.detach())}
