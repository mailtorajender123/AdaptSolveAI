import argparse
import copy
import os
import yaml
import torch
torch.set_num_threads(2)
import pandas as pd

from envs.synthetic_jobshop_env import SyntheticJobShopEnv
from models.selfsolvenet import SelfSolveNet
from training.pretrain_ssl import pretrain_ssl
from training.offline_train_policy import train_policy_imitation
from training.online_evaluate import evaluate_online
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from utils.plot_results import make_plots


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_device(cfg):
    if cfg.get('device', 'auto') == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg.get('device')


def build_env(cfg, shift='ID'):
    env_cfg = dict(cfg['environment'])
    env_cfg['shift_level'] = shift
    return SyntheticJobShopEnv(**env_cfg)


def build_model(cfg, env):
    mcfg = cfg['model']
    # Guard against config mismatch.
    state_dim = env.state_dim
    num_actions = env.action_dim
    model = SelfSolveNet(
        state_dim=state_dim,
        latent_dim=mcfg.get('latent_dim', 128),
        hidden_dim=mcfg.get('hidden_dim', 128),
        num_actions=num_actions,
        num_heads=mcfg.get('num_heads', 4),
        adapter_dim=mcfg.get('adapter_dim', 24),
        dropout=mcfg.get('dropout', 0.05),
    )
    return model


def aggregate_and_save(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    raw_path = os.path.join(out_dir, 'episode_results.csv')
    df.to_csv(raw_path, index=False)
    numeric = df.select_dtypes(include='number').columns.tolist()
    summary = df.groupby(['baseline', 'shift'])[numeric].agg(['mean', 'std']).reset_index()
    summary_path = os.path.join(out_dir, 'summary_results.csv')
    summary.to_csv(summary_path, index=False)
    return raw_path, summary_path


def main():
    parser = argparse.ArgumentParser(description='AdaptSolveAI/SelfSolveNet implementation for online scheduling.')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--quick', action='store_true', help='Run a short smoke-test experiment.')
    parser.add_argument('--skip_train', action='store_true', help='Load checkpoint if available and skip pretraining/policy training.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg['experiment']['pretrain_steps'] = 5
        cfg['experiment']['train_steps'] = 10
        cfg['experiment']['eval_episodes'] = 1
        cfg['environment']['episode_length'] = 20
        cfg['policy_training']['batch_size'] = 16
        cfg['adaptation']['batch_size'] = 8
        cfg['adaptation']['min_buffer_before_adapt'] = 8
        cfg['experiment']['shift_levels'] = ['ID', 'S2']
        cfg['experiment']['baselines'] = ['no_adapt', 'adaptsolveai']

    set_seed(cfg.get('seed', 42))
    device = get_device(cfg)
    os.makedirs(cfg.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    os.makedirs(cfg.get('output_dir', 'results'), exist_ok=True)
    logger = ExperimentLogger(cfg.get('output_dir', 'results'))
    logger.log({'event': 'start', 'device': device})

    env = build_env(cfg, 'ID')
    model = build_model(cfg, env).to(device)
    ckpt_path = os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'selfsolvenet_scheduling.pt')

    if args.skip_train and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.log({'event': 'checkpoint_loaded', 'path': ckpt_path})
    else:
        ssl_logs = pretrain_ssl(model, env, cfg['ssl'],
                                steps=cfg['experiment'].get('pretrain_steps', 400),
                                batch_size=cfg['policy_training'].get('batch_size', 128),
                                lr=cfg['policy_training'].get('lr', 8e-4), device=device)
        logger.log({'event': 'ssl_pretrain_done', 'last': ssl_logs[-1] if ssl_logs else {}})
        pol_logs = train_policy_imitation(model, env,
                                          steps=cfg['experiment'].get('train_steps', 2500),
                                          batch_size=cfg['policy_training'].get('batch_size', 128),
                                          lr=cfg['policy_training'].get('lr', 8e-4), device=device)
        logger.log({'event': 'policy_train_done', 'last': pol_logs[-1] if pol_logs else {}})
        torch.save(model.state_dict(), ckpt_path)
        logger.log({'event': 'checkpoint_saved', 'path': ckpt_path})

    rows = []
    for baseline in cfg['experiment'].get('baselines', ['no_adapt', 'adaptsolveai']):
        for shift in cfg['experiment'].get('shift_levels', ['ID', 'S1', 'S2', 'S3']):
            eval_model = copy.deepcopy(model).to(device)
            eval_env = build_env(cfg, shift)
            logger.log({'event': 'eval_start', 'baseline': baseline, 'shift': shift})
            summaries = evaluate_online(eval_model, eval_env, cfg, baseline=baseline, shift_level=shift,
                                        episodes=cfg['experiment'].get('eval_episodes', 20), device=device)
            rows.extend(summaries)
            mean_return = sum(s['return'] for s in summaries) / max(1, len(summaries))
            mean_svr = sum(s['safety_violation_rate'] for s in summaries) / max(1, len(summaries))
            logger.log({'event': 'eval_done', 'baseline': baseline, 'shift': shift,
                        'mean_return': mean_return, 'mean_safety_violation_rate': mean_svr})

    raw_path, summary_path = aggregate_and_save(rows, cfg.get('output_dir', 'results'))
    plot_paths = make_plots(raw_path, cfg.get('output_dir', 'results'))
    logger.log({'event': 'finished', 'raw_results': raw_path, 'summary_results': summary_path, 'plots': plot_paths})
    logger.save('run_log.json')

if __name__ == '__main__':
    main()
