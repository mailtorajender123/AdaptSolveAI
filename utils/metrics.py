import numpy as np


def confidence_interval_95(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return float(arr.mean()) if len(arr) else 0.0, 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def summarize_episode(records):
    rewards = np.array([r.get('reward', 0.0) for r in records], dtype=float)
    violations = np.array([r.get('safety_violation', 0.0) for r in records], dtype=float)
    deadlines = np.array([r.get('deadline_miss', 0.0) for r in records], dtype=float)
    latencies = np.array([r.get('latency_ms', 0.0) for r in records], dtype=float)
    adapt_lat = np.array([r.get('adapt_latency_ms', 0.0) for r in records], dtype=float)
    tardiness = np.array([r.get('tardiness', 0.0) for r in records], dtype=float)
    throughput = np.array([r.get('completed_jobs', 0.0) for r in records], dtype=float)
    return {
        'return': float(rewards.sum()),
        'avg_reward': float(rewards.mean()) if len(rewards) else 0.0,
        'success_rate': float(np.mean(rewards > -2.0)) if len(rewards) else 0.0,
        'safety_violation_rate': float(violations.mean()) if len(violations) else 0.0,
        'constraint_satisfaction_rate': 1.0 - (float(violations.mean()) if len(violations) else 0.0),
        'deadline_miss_rate': float(deadlines.mean()) if len(deadlines) else 0.0,
        'avg_latency_ms': float(latencies.mean()) if len(latencies) else 0.0,
        'p95_latency_ms': float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        'avg_adapt_latency_ms': float(adapt_lat.mean()) if len(adapt_lat) else 0.0,
        'avg_tardiness': float(tardiness.mean()) if len(tardiness) else 0.0,
        'throughput': float(throughput.sum()),
    }
