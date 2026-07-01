import os
import pandas as pd
import matplotlib.pyplot as plt


def make_plots(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    created = []
    for metric in ['return', 'safety_violation_rate', 'deadline_miss_rate', 'avg_latency_ms']:
        plt.figure(figsize=(8, 5))
        pivot = df.groupby(['baseline', 'shift'])[metric].mean().reset_index()
        for baseline, sub in pivot.groupby('baseline'):
            sub = sub.sort_values('shift')
            plt.plot(sub['shift'], sub[metric], marker='o', label=baseline)
        plt.xlabel('Shift Level')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(metric.replace('_', ' ').title())
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, f'{metric}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        created.append(path)
    return created
