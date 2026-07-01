import numpy as np
import torch
import torch.nn.functional as F


def collect_oracle_dataset(env, n_steps=2500):
    states, actions = [], []
    s = env.reset('ID')
    for _ in range(n_steps):
        a = env.edf_oracle_action()
        states.append(s); actions.append(a)
        ns, _, done, _ = env.step(a)
        s = env.reset('ID') if done else ns
    return np.asarray(states, np.float32), np.asarray(actions, np.int64)


def train_policy_imitation(model, env, steps=2500, batch_size=128, lr=8e-4, device='cpu'):
    states, actions = collect_oracle_dataset(env, n_steps=max(steps, batch_size*4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    logs = []
    n = len(states)
    model.train()
    for step in range(steps):
        idx = np.random.choice(n, size=batch_size, replace=n < batch_size)
        x = torch.tensor(states[idx], dtype=torch.float32, device=device)
        y = torch.tensor(actions[idx], dtype=torch.long, device=device)
        out = model(x)
        ce = F.cross_entropy(out['logits'], y)
        value_loss = 0.001 * out['value'].pow(2).mean()
        loss = ce + value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            acc = (out['logits'].argmax(-1) == y).float().mean().item()
            logs.append({'step': step, 'policy_loss': float(loss.detach()), 'imit_acc': acc})
    return logs
