"""Checkpoint 维度迁移 / 热启动工具。

将旧 checkpoint 的网络参数迁移到新的观测维度（例如修改 height_scan 采样、
增删观测项之后），并重建与 trainer 完全一致的 optax 优化器状态，
使得 ``train.py --resume`` 可以直接加载续训（actor 热启动 + critic 重估）。

设计要点（与旧版脚本的区别）：
- 不迁移旧 opt_state，而是用 trainer 相同的 optimizer 定义对新 params 重新
  ``init``。Adam 的二阶矩对形状变化的层没有可迁移价值，重建可彻底避免
  形状不匹配，且未变化层的损失仅是初期收敛略慢。
- 参数拷贝按“结构 + 形状”匹配，无任何硬编码维度：同名同形状直接拷贝；
  仅输入维度（第一维）变化的层做前缀拷贝（旧权重放前部，新行保留新初始化）；
  其余保留新初始化并在报告中列出。
- 支持传入 ``transforms`` 自定义某个参数叶的迁移方式，例如按空间位置将旧
  height_scan 输入行重采样到新网格::

      transforms = {
          "params/critic_mlp_0/kernel": lambda old_w: resample_scan_rows(old_w, ...)
      }

用法（在 code/ 目录下，新维度的观测需先在 cfg 中改好）::

    python src/migrate_checkpoint.py --old <旧ckpt.pkl> --new <新ckpt.pkl> [--exp xg_velocity]
"""

from __future__ import annotations

import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(path, checkpoint):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def build_optimizer(learning_rate: float, max_grad_norm: float):
    """与 training/ppo.py 中 trainer 的 optimizer 定义保持一致。"""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate, eps=1e-5),
    )


def copy_compatible_leaves(old_params, new_params, transforms=None):
    """把旧参数树中兼容的叶子并入新参数树。

    匹配规则（按展平后的参数路径）：
    - ``transforms`` 中提供了自定义函数的叶子：用 ``fn(old_leaf)`` 的结果；
    - 新旧形状完全一致：拷贝旧叶子；
    - 仅第一维不同（输入层维度变化）且其余维度一致：前缀拷贝（截断或填充），
      新行保留新初始化；
    - 其余：保留新初始化。

    Returns:
        (merged_params, report): report 为 [(path, old_shape, new_shape, mode)]。
    """
    transforms = transforms or {}
    old_flat = traverse_util.flatten_dict(old_params)
    new_flat = traverse_util.flatten_dict(new_params)

    out = {}
    report = []
    for key, new_val in new_flat.items():
        path = "/".join(str(k) for k in key)
        old_val = old_flat.get(key)

        if path in transforms:
            if old_val is None:
                raise KeyError(f"transform 提供了不存在的参数路径: {path}")
            out[key] = jnp.asarray(transforms[path](np.asarray(old_val)))
            report.append((path, old_val.shape, out[key].shape, "transform"))
        elif old_val is not None and old_val.shape == new_val.shape:
            out[key] = old_val
            report.append((path, old_val.shape, new_val.shape, "copied"))
        elif (
            old_val is not None
            and old_val.ndim == new_val.ndim
            and old_val.shape[1:] == new_val.shape[1:]
        ):
            rows = min(old_val.shape[0], new_val.shape[0])
            merged = new_val.at[:rows].set(old_val[:rows])
            mode = "padded" if old_val.shape[0] < new_val.shape[0] else "truncated"
            out[key] = merged
            report.append((path, old_val.shape, new_val.shape, mode))
        else:
            out[key] = new_val
            report.append((path, getattr(old_val, "shape", None), new_val.shape, "fresh-init"))

    return traverse_util.unflatten_dict(out), report


def migrate_checkpoint(
    old_path: str,
    new_path: str,
    network,
    obs_size: int,
    critic_obs_size: int,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    transforms=None,
):
    """把 old_path 的 checkpoint 迁移到新观测维度并保存到 new_path。

    Args:
        network: 新维度的 ActorCriticMLP 实例（尚未 init）。
        obs_size / critic_obs_size: 新的 actor / critic 观测维度。
        transforms: {展平参数路径: callable(old_array) -> new_array}，可选。
    """
    old_ckpt = load_checkpoint(old_path)
    old_params = old_ckpt["params"]

    rng = jax.random.PRNGKey(seed)
    new_params = network.init(
        rng,
        jnp.zeros((1, obs_size)),
        jnp.zeros((1, critic_obs_size)),
    )

    merged, report = copy_compatible_leaves(old_params, new_params, transforms)

    modes = {}
    for path, old_shape, new_shape, mode in report:
        modes[mode] = modes.get(mode, 0) + 1
        if mode != "copied":
            print(f"  [{mode:9s}] {path}: {old_shape} -> {new_shape}")
    print(f"  共 {len(report)} 个参数叶: " + ", ".join(f"{k}={v}" for k, v in sorted(modes.items())))

    # 重建与 trainer 完全一致的优化器状态（形状与新 params 天然匹配）
    optimizer = build_optimizer(learning_rate, max_grad_norm)
    new_opt_state = optimizer.init(merged)

    new_ckpt = {
        "params": merged,
        "opt_state": new_opt_state,
        "step": 0,
        "rng": old_ckpt.get("rng", np.asarray(rng)),
        "iteration": 0,
    }
    save_checkpoint(new_path, new_ckpt)
    print(f"已保存迁移后的 checkpoint: {new_path}")
    print("用 train.py --resume 加载续训即可（iteration 从 0 重新计数）。")
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(description="将 checkpoint 迁移到新观测维度（热启动）")
    parser.add_argument("--old", required=True, help="旧 checkpoint 路径")
    parser.add_argument("--new", required=True, help="迁移后保存路径")
    parser.add_argument("--exp", default="xg_velocity", help="experiment 名称（决定网络结构与维度）")
    parser.add_argument("--actor_obs", type=int, default=None, help="覆盖 actor 观测维度（默认读 cfg.env.num_observations）")
    parser.add_argument("--critic_obs", type=int, default=None, help="覆盖 critic 观测维度（默认读 cfg.robot.num_privileged_obs）")
    args = parser.parse_args()

    from cfg.experiments import get_experiment
    from networks import ActorCriticMLP

    spec = get_experiment(args.exp)
    env_cfg = spec.build_env_config()
    ppo_cfg = spec.build_ppo_config()

    obs_size = args.actor_obs or env_cfg.env.num_observations
    critic_obs_size = args.critic_obs or env_cfg.robot.num_privileged_obs

    network = ActorCriticMLP(
        action_size=env_cfg.env.num_actions,
        actor_hidden_dims=tuple(ppo_cfg.network.policy_hidden_layer_sizes),
        critic_hidden_dims=tuple(ppo_cfg.network.value_hidden_layer_sizes),
        init_noise_std=ppo_cfg.network.init_noise_std,
        activation=ppo_cfg.network.activation,
    )

    print(f"迁移: actor_obs {env_cfg.env.num_observations}->{obs_size}, "
          f"critic_obs {env_cfg.robot.num_privileged_obs}->{critic_obs_size}")
    migrate_checkpoint(
        args.old,
        args.new,
        network,
        obs_size,
        critic_obs_size,
        learning_rate=ppo_cfg.learning_rate,
        max_grad_norm=ppo_cfg.max_grad_norm,
    )


if __name__ == "__main__":
    main()
