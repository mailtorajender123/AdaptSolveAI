import copy
import numpy as np
import torch


def collect_random_transitions(env, n_steps=1000):
    states, next_states, margins = [], [], []
    s = env.reset()
    for _ in range(n_steps):
        a = env.edf_oracle_action() if env.rng.random() < 0.7 else int(env.rng.integers(0, env.action_dim))
        margin = 1.0
        if env.queue:
            job = env.queue[0]
            slack = job.deadline - env.t
            projected = np.min(env.machine_remaining + job.processing_time)
            margin = float((slack - projected) / max(1.0, slack))
        ns, r, done, info = env.step(a)
        states.append(s); next_states.append(ns); margins.append(margin)
        s = env.reset() if done else ns
    return np.asarray(states, np.float32), np.asarray(next_states, np.float32), np.asarray(margins, np.float32)


def pretrain_ssl(model, env, ssl_cfg, steps=400, batch_size=128, lr=1e-3, device='cpu'):
    states, next_states, margins = collect_random_transitions(env, max(steps * 2, batch_size * 4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    model.train()
    logs = []
    n = len(states)
    for step in range(steps):
        idx = np.random.choice(n, size=batch_size, replace=n < batch_size)
        batch = {
            'state': torch.tensor(states[idx], dtype=torch.float32, device=device),
            'next_state': torch.tensor(next_states[idx], dtype=torch.float32, device=device),
            'safety_margin': torch.tensor(margins[idx], dtype=torch.float32, device=device),
        }
        loss, parts = model.compute_ssl_loss(batch, ssl_cfg, teacher=teacher)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            logs.append({'step': step, **parts})
    return logs
