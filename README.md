# MJX_ZS

A reinforcement-learning locomotion training framework for quadruped robots, built on Google MJX (MuJoCo XLA).

## Project Structure

```
├── src/
│   ├── train.py                 # Experiment-registry-driven training entrypoint
│   ├── play.py                  # Policy visualization and inference script
│   ├── cfg/
│   │   ├── xg_config.py         # Environment and PPO hyperparameter config
│   │   └── experiments/         # Experiment registry (xg_velocity)
│   ├── components/
│   │   ├── base.py              # BaseComponent — 7 task abstract methods
│   │   ├── mechanical_base.py   # MechanicalBase — default mechanical implementations
│   │   ├── factory.py           # create_component() component factory
│   │   ├── subcomponents/       # Reusable subcomponents (control, feet, history, ...)
│   │   └── tasks/
│   │       └── velocity/        # Velocity tracking task implementation
│   ├── envs/
│   │   ├── base.py              # BaseEnv abstract interface
│   │   ├── state.py             # State data structure (with EnvInfo)
│   │   ├── info.py              # EnvInfo typed schema (flax.struct)
│   │   ├── info_compat.py       # EnvInfo ↔ dict bridge layer
│   │   ├── factory.py           # build_env() environment factory
│   │   └── backends/
│   │       ├── generic_env.py   # Generic MJX environment backend
│   │       └── viewer.py        # MuJoCo visualization mixin
│   ├── networks/                # Actor-Critic networks (MLP / LSTM / OdomLSTM)
│   ├── robots/                  # Robot specifications (XG)
│   ├── tasks/                   # Task specification metadata
│   └── training/
│       └── ppo.py               # PPO trainer
├── assets/
│   ├── bot_model/xg/            # Robot URDF/MJCF model and meshes
│   └── nn_model/mlp/            # Pre-trained model weights
├── scripts/
│   ├── train.sh                 # Training launcher
│   └── play.sh                  # Inference launcher
├── doc/                         # Architecture documentation and diagrams
├── checkpoints/                 # Training checkpoint output directory
└── requirements.txt
```

## Key Features

- **Pure MLP network**: Actor-Critic architecture `(512, 256, 128)`, ELU activation, no LSTM.
- **Asymmetric Actor-Critic**: Actor uses 48D observations; critic uses 72D privileged observations.
- **PPO training**: GAE advantage estimation, clipped value function, adaptive learning rate.
- **Component-based architecture**: Task logic decoupled from simulation mechanics via `BaseComponent` / `MechanicalBase`; new tasks only need to implement 7 methods.
- **Typed state**: `EnvInfo` (flax.struct) replaces bare `dict`, JAX pytree compatible.
- **Domain randomization**: motor strength, Kp/Kd factor, dof damping, friction, mass, COM offset, inertia, restitution, action/torque/obs latency, and random push.
- **Real-time visualization**: MuJoCo viewer and video export supported.

## Quick Start

### Environment Setup

```bash
conda create -n MJX python=3.10 -y
conda activate MJX
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Requires an NVIDIA GPU and CUDA 12.

### Training

```bash
# Use the launcher script
bash scripts/train.sh

# Or call the experiment-registry entrypoint directly
python src/train.py --exp xg_velocity --max_iteration 5000 --num_envs 4096
```

Main parameters:
- `--max_iteration`: Number of training iterations. When omitted, falls back to `config.max_iterations = 10000` in `src/cfg/xg_config.py`.
- `--num_envs`: Number of parallel environments. When omitted, falls back to `config.env.num_envs = 4096` in `src/cfg/xg_config.py`.
- `--seed`: Training RNG seed (default 42).
- `--init_noise_std`: Override the actor's initial action noise std.
- `--resume`: Path to a checkpoint to resume training from.

### Inference / Visualization

```bash
# Use the pre-trained model bundled in assets/
bash scripts/play.sh

# Or run directly (mirrors scripts/play.sh)
python src/play.py --exp xg_velocity --checkpoint assets/nn_model/mlp/iteration_1800.pkl

# Save a video (offscreen render, no display required)
python src/play.py --exp xg_velocity --checkpoint assets/nn_model/mlp/iteration_1800.pkl --save_video

# Specify a velocity command
python src/play.py --exp xg_velocity --checkpoint assets/nn_model/mlp/iteration_1800.pkl --cmd_vel_x 1.0 --cmd_vel_y 0.0
```

Main parameters:
- `--exp`: Experiment registered in `cfg/experiments/` (default `xg_velocity`).
- `--checkpoint`: Path to `iteration_*.pkl` (or its directory). Defaults to the latest run under `checkpoints/<exp>/`.
- `--random_policy`: Use the random init instead of loading a checkpoint.
- `--device`: JAX backend (`cpu` / `cuda` / `gpu` / `tpu`). Default auto-detect (GPU if available).
- `--deterministic`: Use action mean (no Gaussian sampling). Default `True`.
- `--num_envs`, `--episode_length`, `--seed`: Rollout sizing and RNG seed.
- `--cmd_vel_x/y/yaw`: Pin the velocity command for the rollout.
- `--num_render_envs`: Robots in the multi-robot grid.
- `--save_video`: Render offscreen to a video file instead of the live viewer.
- `--video_path`: Output video path (default `play_output.mp4`).
- `--render_every`, `--width`, `--height`, `--fps`: Video rendering controls.

## Configuration

The core configuration lives in `src/cfg/xg_config.py` and is split into two parts:

| Config | Description |
|------|------|
| `get_xg_config()` | Environment config: observation space, rewards, domain randomization, termination conditions, etc. |
| `get_xg_ppo_config()` | PPO config: learning rate, network architecture, training hyperparameters, etc. |

## Architecture Overview

The project uses a **layered, component-based architecture** that decouples task logic from simulation mechanics:

```
BaseComponent (ABC)          — 7 task abstract methods (command/observation/reward/termination)
  └─ MechanicalBase          — default mechanical implementations (physics/history/contact/reset)
       └─ VelocityComponent  — concrete implementation of the XG velocity tracking task
```

- **GenericEnv** loads task components through the `create_component()` factory and only owns the MJX data lifecycle and `step()` orchestration.
- **EnvInfo** (`flax.struct.dataclass`) is a typed state bus that replaces the previous `dict[str, Any]`.
- **9 subcomponents** (control, domain_randomization, feet, history, kinematics, perturbation, physics, reset, topology) encapsulate reusable mechanical logic.

See `doc/ARCHITECTURE.md` for the detailed design.

### Adding a New Task

To add a new task, you only need to:

1. Create a new directory under `components/tasks/`.
2. Inherit from `MechanicalBase` and implement the 7 task methods.
3. Register it in `_TASK_REGISTRY` inside `components/tasks/__init__.py`.
4. Add a new experiment description under `cfg/experiments/`.

## Tech Stack

- **Simulation**: MuJoCo MJX (JAX-accelerated).
- **Training**: JAX + Flax + Optax (PPO).
- **Model**: XG quadruped robot (12 degrees of freedom).
- **Logging**: TensorBoard.
