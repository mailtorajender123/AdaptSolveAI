#  AdaptSolveAI
### Adaptive Self-Supervised Learning for Real-Time Problem Solving in Autonomous Systems

---

##  Overview

**AdaptSolveAI** is a modular PyTorch implementation of an adaptive self-supervised learning framework for real-time autonomous decision-making under distribution shifts.

The framework implements the proposed **SelfSolveNet** architecture with:

- Multi-objective self-supervised learning
- Shift-aware prioritized replay
- Latency-bounded online adaptation
- Safety-supervised action filtering

The current implementation focuses on a fully reproducible synthetic online job-shop scheduling environment, enabling rapid experimentation without requiring computationally expensive simulators such as CARLA, Habitat, or ManiSkill2.

The architecture is modular and can easily be extended to additional autonomous system domains through environment wrappers and modality-specific encoders.

---

#  Key Features

-  Complete PyTorch implementation of **SelfSolveNet**
-  Synthetic online job-shop scheduling environment
-  Multi-modal encoder and fusion architecture
-  Multi-objective Self-Supervised Learning

Supported SSL objectives include:

- Contrastive Predictive Coding (CPC)
- Masked Autoencoder (MAE) Reconstruction
- Teacher–Student Distillation
- Safety-aware Representation Learning

Additional capabilities include:

- Policy head for real-time decision making
- Distribution shift detection
- Shift-aware prioritized replay memory
- Latency-constrained online adaptation
- Adapter-based parameter-efficient fine-tuning
- Elastic Weight Consolidation (EWC)
- Runtime safety supervisor
- Baseline and ablation-ready implementation
- Automatic logging and visualization

---

#  Project Structure

```text
AdaptSolveAI_Code/
│
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── envs/
│   └── synthetic_jobshop_env.py
│
├── models/
│   ├── selfsolvenet.py
│   ├── encoders.py
│   ├── fusion.py
│   ├── ssl_heads.py
│   ├── policy_head.py
│   └── adapter_layers.py
│
├── adaptation/
│   ├── shift_detector.py
│   ├── prioritized_replay.py
│   ├── online_adapter.py
│   └── ewc.py
│
├── safety/
│   ├── cbf_filter.py
│   └── safety_metrics.py
│
├── training/
│   ├── pretrain_ssl.py
│   ├── offline_train_policy.py
│   └── online_evaluate.py
│
├── utils/
│   ├── metrics.py
│   ├── logger.py
│   ├── seed.py
│   ├── latency_profiler.py
│   └── plot_results.py
│
├── checkpoints/
├── figures/
└── results/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/AdaptSolveAI.git

cd AdaptSolveAI
```

Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyYAML
- tqdm

> GPU is recommended for faster training, although the synthetic scheduling environment can run efficiently on CPU.

---

#  Quick Start

Run the complete pipeline:

```bash
python main.py
```

This automatically performs:

1. Synthetic environment creation
2. SelfSolveNet initialization
3. Offline SSL pretraining
4. Policy training
5. Online evaluation
6. Distribution shift adaptation
7. Safety filtering
8. Result logging
9. Figure generation

---

#  Generated Outputs

```
results/
│
├── metrics.csv
├── episode_logs.csv

figures/
│
├── reward_curve.png
├── latency_curve.png
└── safety_violation_curve.png

checkpoints/
│
├── selfsolvenet_pretrained.pt
└── selfsolvenet_final.pt
```

---

#  Configuration

Experiment parameters are stored inside:

```text
config.yaml
```

Example configuration:

```yaml
seed: 42
device: auto

env:
  num_machines: 5
  max_queue_size: 20
  episode_length: 300
  shift_type: bursty

model:
  state_dim: 16
  latent_dim: 128
  hidden_dim: 256
  num_actions: 5

training:
  ssl_epochs: 10
  policy_epochs: 10
  batch_size: 64
  learning_rate: 0.0003

adaptation:
  enabled: true
  max_steps: 3
  latency_budget_ms: 3
  replay_capacity: 10000
  adapter_lr: 0.001

safety:
  enable_filter: true
  max_machine_utilization: 1.0
  deadline_slack_margin: 0.05
```

---

# Method Overview

AdaptSolveAI follows a continuous

```
Observation
      ↓
State Encoding
      ↓
Shared Representation
      ↓
Policy Generation
      ↓
Safety Supervisor
      ↓
Action Execution
      ↓
Distribution Shift Detection
      ↓
Online Adaptation
```

Unsafe actions are replaced with feasible alternatives before execution.

---

# SelfSolveNet Architecture

The proposed SelfSolveNet consists of:

- State Encoder
- Shared Fusion Layer
- Self-Supervised Learning Heads
- Policy Head
- Adapter Modules

---

#  Self-Supervised Learning Objectives

The total optimization objective is

\[
L_{total}=
\lambda_{cpc}L_{cpc}
+\lambda_{mae}L_{mae}
+\lambda_{dist}L_{dist}
+\lambda_{safety}L_{safety}
+\lambda_{ewc}L_{ewc}
\]

Where

| Loss | Description |
|-------|-------------|
| **L_CPC** | Temporal consistency learning |
| **L_MAE** | Masked feature reconstruction |
| **L_Dist** | Teacher–student distillation |
| **L_Safety** | Safety-aware latent learning |
| **L_EWC** | Catastrophic forgetting prevention |

---

#  Online Adaptation

Online adaptation is triggered whenever

```text
shift_score > shift_threshold
```

or

```text
policy_entropy > entropy_threshold
```

The adaptation module

- Updates only lightweight adapter parameters
- Uses prioritized replay sampling
- Respects latency constraints
- Immediately deploys the adapted model

---

# Shift-Aware Prioritized Replay

Replay priority is computed using

- Latent novelty
- Safety boundary proximity
- Fisher-based parameter importance

Higher priority samples are replayed more frequently.

---

# Safety Supervisor

The runtime safety supervisor verifies every proposed action.

If safe:

```text
Execute policy action
```

Otherwise:

```text
Replace with nearest feasible action
```

This approximates a Control Barrier Function (CBF)-based runtime safety layer.

---

# Evaluation Metrics

The implementation reports:

- Episode Reward
- Success Rate
- Normalized Return
- Adaptation Gain
- Safety Violation Rate
- Constraint Satisfaction Rate
- Deadline Miss Rate
- Average Tardiness
- Throughput
- Machine Utilization
- Average Decision Latency
- P95 Latency
- Adaptation Trigger Count

---

# Running Baselines

Example:

```bash
python experiments/run_no_adapt.py

python experiments/run_adabn.py

python experiments/run_entropy_min.py

python experiments/run_distill_only.py

python experiments/run_adaptsolveai.py
```

Supported baselines

- NoAdapt
- AdaBN
- EntropyMin
- DistillOnly
- Online RL
- AdaptSolveAI

---

# Ablation Studies

Recommended experiments

- Without CPC
- Without MAE
- Without Distillation
- Without Safety Loss
- Without Prioritized Replay
- Uniform Replay
- Without Online Adaptation
- Without Safety Supervisor
- Different Latency Budgets
- Different Replay Buffer Sizes

---

# Extending to Other Domains

The framework can be extended to

- CARLA Autonomous Driving
- Habitat Indoor Navigation
- ManiSkill2 Robotic Manipulation

Simply implement

- Environment Wrapper
- Domain-specific Encoders
- Appropriate Action Space

---

# Reproducibility

The project supports

- Fixed random seeds
- Configurable experiments
- Automatic checkpointing
- CSV logging
- Figure generation
- Deterministic synthetic environments

---

#  Citation

```bibtex
@article{adaptsolveai2026,
  title={Adaptive Self-Supervised Learning for Real-Time Problem Solving in Autonomous Systems},
  author={mailtorajender},
  journal={To be updated},
  year={2026}
}
```

---

#  License

This project is released for academic and research purposes.

Recommended License:

**MIT License**

---

#  Contributing

Contributions are welcome!

Feel free to fork the repository, create a feature branch, and submit a pull request.



---
