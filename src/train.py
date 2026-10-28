"""Experiment-driven training CLI."""

from __future__ import annotations

import argparse
import os
from datetime import datetime


def _configure_xla():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_triton_gemm_any=True" not in xla_flags:
        xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags


_configure_xla()

from cfg.experiments import get_experiment, list_experiments
from envs import build_env
from networks import ActorCriticMLP
from training.ppo import train


def main():
    parser = argparse.ArgumentParser(description="Train MJX legged robot experiments")
    parser.add_argument("--exp", default="xg_velocity", choices=list_experiments())
    parser.add_argument("--max_iteration", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init_noise_std", type=float, default=None)
    args = parser.parse_args()

    spec = get_experiment(args.exp)
    env_cfg = spec.build_env_config()
    ppo_cfg = spec.build_ppo_config()

    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
        ppo_cfg.num_envs = args.num_envs
    if args.max_iteration is not None:
        ppo_cfg.max_iterations = args.max_iteration
    if args.init_noise_std is not None:
        ppo_cfg.network.init_noise_std = args.init_noise_std

    env = build_env(spec, env__num_envs=env_cfg.env.num_envs)
    network = ActorCriticMLP(
        action_size=env.action_size,
        actor_hidden_dims=tuple(ppo_cfg.network.policy_hidden_layer_sizes),
        critic_hidden_dims=tuple(ppo_cfg.network.value_hidden_layer_sizes),
        init_noise_std=ppo_cfg.network.init_noise_std,
        activation=ppo_cfg.network.activation,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"checkpoints/{args.exp}/tensorboard_{timestamp}"
    checkpoint_dir = f"checkpoints/{args.exp}/checkpoints_{timestamp}"

    train(
        env=env,
        network=network,
        num_envs=env_cfg.env.num_envs,
        num_steps_per_env=ppo_cfg.unroll_length,
        num_learning_epochs=ppo_cfg.num_updates_per_batch,
        num_mini_batches=ppo_cfg.num_minibatches,
        learning_rate=ppo_cfg.learning_rate,
        gamma=ppo_cfg.discounting,
        lam=ppo_cfg.gae_lambda,
        clip_param=ppo_cfg.clip_param,
        vf_coef=ppo_cfg.value_loss_coef,
        entropy_coef=ppo_cfg.entropy_cost,
        max_grad_norm=ppo_cfg.max_grad_norm,
        max_iterations=ppo_cfg.max_iterations,
        desired_kl=ppo_cfg.desired_kl,
        seed=args.seed,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        save_interval=ppo_cfg.save_interval,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
