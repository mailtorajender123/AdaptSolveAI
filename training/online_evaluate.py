import time
import numpy as np
import torch
from adaptation.shift_detector import ShiftEntropyDetector
from adaptation.prioritized_replay import ShiftAwarePrioritizedReplay
from adaptation.online_adapter import LatencyBoundedAdapter
from safety.cbf_filter import SchedulingSafetySupervisor
from utils.metrics import summarize_episode
from models.adapter_layers import freeze_except_adapters_bn, unfreeze_all


def calibrate_detector(model, env, detector, n_steps=400, device='cpu', quantile=0.90):
    s = env.reset('ID')
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            a, lp, ent, out = model.act(x, deterministic=False)
            detector.update_stats(out['z'], ent.item())
            ns, _, done, _ = env.step(int(a.item()))
            s = env.reset('ID') if done else ns
    detector.calibrate_entropy(quantile)


def evaluate_online(model, env, cfg, baseline='adaptsolveai', shift_level='S1', episodes=10, device='cpu'):
    model = model.to(device)
    model.eval()
    safety = SchedulingSafetySupervisor(cfg['safety'].get('fallback', 'least_loaded'))
    detector = ShiftEntropyDetector(cfg['model']['latent_dim'],
                                    shift_threshold=cfg['adaptation'].get('shift_threshold', 3.0))
    calibrate_detector(model, env, detector, device=device,
                       quantile=cfg['adaptation'].get('trigger_entropy_quantile', 0.90))
    replay = ShiftAwarePrioritizedReplay(
        capacity=cfg['adaptation'].get('buffer_capacity', 10000),
        alpha=cfg['adaptation'].get('priority_alpha', 0.6),
        novelty_w=cfg['adaptation'].get('priority_novelty_weight', 0.5),
        safety_w=cfg['adaptation'].get('priority_safety_weight', 0.3),
        fisher_w=cfg['adaptation'].get('priority_fisher_weight', 0.2),
        seed=cfg.get('seed', 42)
    )
    adapter = None
    if baseline in ['adaptsolveai', 'distill_only', 'entropy_min', 'adabn']:
        if baseline == 'adabn':
            # BN/LayerNorm/adapters update through SSL, approximating statistics-only adaptation.
            pass
        adapter = LatencyBoundedAdapter(model, replay, cfg['ssl'], cfg['adaptation'], device=device)
    episode_summaries = []
    for ep in range(episodes):
        s = env.reset(shift_level)
        records = []
        done = False
        while not done:
            model.eval()
            t0 = time.perf_counter()
            x = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            feasible = env.feasible_actions()
            with torch.no_grad():
                deterministic = baseline != 'online_rl'
                action, logp, entropy, out = model.act(x, deterministic=deterministic, feasible_mask=None)
                proposed = int(action.item())
                z = out['z']
            if baseline == 'no_adapt':
                final_action = proposed
                safety_info = {'modified': 0, 'safety_violation': 0}
            else:
                final_action, safety_info = safety.filter_action(env, proposed)
            margin = safety.margin(env)
            ns, reward, done, info = env.step(final_action)
            trigger, det_info = detector.should_adapt(z, entropy)
            detector.update_stats(z, entropy.item())
            replay.add(s, final_action, reward, ns, done, safety_margin=margin,
                       novelty=det_info['shift_score'], fisher=0.0)
            adapt_info = {'adapted': 0, 'steps': 0, 'adapt_latency_ms': 0.0, 'loss': 0.0}
            if baseline == 'adaptsolveai' and trigger:
                adapt_info = adapter.adapt()
            elif baseline == 'distill_only' and trigger:
                old = dict(cfg['ssl'])
                cfg['ssl']['cpc_weight'] = 0.0; cfg['ssl']['mae_weight'] = 0.0; cfg['ssl']['safety_weight'] = 0.0; cfg['ssl']['distill_weight'] = 1.0
                adapt_info = adapter.adapt()
                cfg['ssl'].update(old)
            elif baseline == 'entropy_min' and trigger:
                # Lightweight entropy-minimization update using latest state only.
                if len(replay) >= cfg['adaptation'].get('min_buffer_before_adapt', 64):
                    start = time.perf_counter()
                    params = freeze_except_adapters_bn(model)
                    opt = torch.optim.AdamW(params, lr=cfg['adaptation'].get('lr', 6e-4))
                    batch = replay.sample(cfg['adaptation'].get('batch_size', 32), device)
                    logits = model(batch['state'])['logits']
                    probs = torch.softmax(logits, -1)
                    loss = (probs * torch.log(probs.clamp_min(1e-8))).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    adapt_info = {'adapted': 1, 'steps': 1, 'adapt_latency_ms': (time.perf_counter()-start)*1000, 'loss': float(loss.detach())}
            elif baseline == 'adabn' and trigger:
                # Forward pass with train mode updates BN running stats without gradient.
                if len(replay) >= cfg['adaptation'].get('min_buffer_before_adapt', 64):
                    start = time.perf_counter(); model.train()
                    with torch.no_grad():
                        _ = model(replay.sample(cfg['adaptation'].get('batch_size', 32), device)['state'])
                    model.eval()
                    adapt_info = {'adapted': 1, 'steps': 1, 'adapt_latency_ms': (time.perf_counter()-start)*1000, 'loss': 0.0}
            latency_ms = (time.perf_counter() - t0) * 1000.0
            records.append({
                'reward': reward,
                'safety_violation': max(info.get('safety_violation', 0), safety_info.get('safety_violation', 0)),
                'deadline_miss': info.get('deadline_miss', 0),
                'tardiness': info.get('tardiness', 0.0),
                'completed_jobs': info.get('completed_jobs', 0),
                'latency_ms': latency_ms,
                'adapt_latency_ms': adapt_info['adapt_latency_ms'],
                'trigger': int(trigger),
                'adapted': adapt_info['adapted'],
                **det_info,
            })
            s = ns
        summary = summarize_episode(records)
        summary.update({'baseline': baseline, 'shift': shift_level, 'episode': ep,
                        'trigger_count': int(sum(r['trigger'] for r in records)),
                        'adapt_count': int(sum(r['adapted'] for r in records))})
        episode_summaries.append(summary)
    unfreeze_all(model)
    return episode_summaries
