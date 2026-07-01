"""Simple ablation launcher. Edit the SSL weights below or pass a modified config."""
import os
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
for name, zero_key in [('no_cpc', 'cpc_weight'), ('no_mae', 'mae_weight'), ('no_distill', 'distill_weight'), ('no_safety', 'safety_weight')]:
    c = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
    c['ssl'][zero_key] = 0.0
    c['output_dir'] = f'results_ablation_{name}'
    tmp = f'config_{name}.yaml'
    with open(tmp, 'w', encoding='utf-8') as f:
        yaml.safe_dump(c, f)
    os.system(f'python main.py --config {tmp}')
