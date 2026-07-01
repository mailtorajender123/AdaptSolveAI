import copy
import time
import torch
from .ewc import EWC
from models.adapter_layers import freeze_except_adapters_bn, unfreeze_all

class LatencyBoundedAdapter:
    def __init__(self, model, replay, ssl_cfg, adapt_cfg, device='cpu'):
        self.model = model
        self.replay = replay
        self.ssl_cfg = ssl_cfg
        self.adapt_cfg = adapt_cfg
        self.device = device
        self.teacher = copy.deepcopy(model).to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.params = freeze_except_adapters_bn(self.model)
        self.optimizer = torch.optim.AdamW(self.params, lr=adapt_cfg.get('lr', 6e-4), weight_decay=1e-6)
        self.ewc = EWC(self.model, beta=adapt_cfg.get('ewc_beta', 0.001))

    def adapt(self):
        if len(self.replay) < self.adapt_cfg.get('min_buffer_before_adapt', 64):
            return {'adapted': 0, 'steps': 0, 'adapt_latency_ms': 0.0, 'loss': 0.0}
        start = time.perf_counter()
        max_steps = int(self.adapt_cfg.get('max_steps', 3))
        budget_ms = float(self.adapt_cfg.get('latency_budget_ms', 3.0))
        batch_size = int(self.adapt_cfg.get('batch_size', 32))
        total_loss = 0.0
        steps = 0
        self.model.train()
        for _ in range(max_steps):
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if elapsed_ms >= budget_ms:
                break
            batch = self.replay.sample(batch_size, self.device)
            loss, _ = self.model.compute_ssl_loss(batch, self.ssl_cfg, teacher=self.teacher)
            loss = loss + self.ewc.penalty(self.model)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        # EMA teacher update
        decay = 0.99
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.model.parameters()):
                tp.data.mul_(decay).add_(sp.data, alpha=1.0-decay)
        self.model.eval()
        return {'adapted': int(steps > 0), 'steps': steps, 'adapt_latency_ms': (time.perf_counter() - start)*1000.0, 'loss': total_loss/max(1, steps)}

    def restore_full_training(self):
        unfreeze_all(self.model)
