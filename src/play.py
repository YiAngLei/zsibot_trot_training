"""Play/Visualize script for the MJX velocity experiments.

Loads a checkpoint produced by ``train.py`` and visualizes
the resulting policy either in the live MuJoCo viewer or by writing a video.
Mirrors ``train.py``'s framework wiring: experiment registry → environment
factory → ``ActorCriticMLP``.

Usage:
    # Live viewer with the latest xg_velocity checkpoint
    python play.py

    # Specific checkpoint + commanded velocity
    python play.py --checkpoint checkpoints/xg_velocity/checkpoints_xxx/iteration_5000.pkl \
                   --cmd_vel_x 1.0 --cmd_vel_y 0.0 --cmd_vel_yaw 0.0

    # Save video (no display required)
    python play.py --save_video --video_path out.mp4

    # Random policy (smoke-test rendering)
    python play.py --random_policy

    # Multi-robot grid
    python play.py --num_render_envs 16
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any


def _configure_jax_device() -> None:
    """Default JAX to auto-detect (GPU if available, else CPU).

    ``setdefault`` respects an explicit ``JAX_PLATFORMS`` from the user's env
    (so ``JAX_PLATFORMS=cpu python play.py ...`` still works), and the
    ``--device`` CLI flag below upgrades this to a hard set when provided.
    """
    pass


def _pre_parse_device() -> None:
    """Honour ``--device`` BEFORE jax is imported.

    argparse runs in ``main()`` long after jax is on the import path, so we
    have to peek at sys.argv ourselves to override the JAX backend in time.
    """
    for i, tok in enumerate(sys.argv[1:], start=1):
        if tok == "--device" and i < len(sys.argv) - 1:
            os.environ["JAX_PLATFORMS"] = sys.argv[i + 1]
            return
        if tok.startswith("--device="):
            os.environ["JAX_PLATFORMS"] = tok.split("=", 1)[1]
            return


_pre_parse_device()
_configure_jax_device()

# Add ``src`` to path so the script works whether invoked as ``python src/play.py``
# or ``cd src && python play.py`` (matches train.py behaviour).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np
from brax import math

from cfg.experiments import get_experiment, list_experiments
from components.tasks.velocity.observations import (
    HEIGHT_SCAN_NUM_X,
    HEIGHT_SCAN_NUM_Y,
    VELOCITY_PRIVILEGED_SCHEMA,
)
from envs import load_symbol
from networks import ActorCriticMLP

try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False


# Dimensionality of the odom signal forwarded to ``env.step``.  Currently the
# env consumes a 3-vector (base linear velocity x/y/z).  Kept as a constant so
# play.py and training/ppo.py stay in lockstep if the schema ever grows.
_ODOM_DIM = 3


# ============================================================================
# Checkpoint helpers
# ============================================================================

def _find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Return the highest ``iteration_*.pkl`` (training.ppo format) in a dir."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    candidates = []
    for name in os.listdir(checkpoint_dir):
        if not name.endswith(".pkl"):
            continue
        if name.startswith("iteration_"):
            try:
                iteration = int(name.replace("iteration_", "").replace(".pkl", ""))
            except ValueError:
                continue
            candidates.append((iteration, name))
        elif name.startswith("checkpoint_"):
            # Backwards-compat with old LSTM checkpoint naming.
            if name == "checkpoint_latest.pkl":
                return os.path.join(checkpoint_dir, name)
            try:
                iteration = int(name.replace("checkpoint_", "").replace(".pkl", ""))
            except ValueError:
                continue
            candidates.append((iteration, name))

    if not candidates:
        raise FileNotFoundError(f"No iteration_*.pkl found in {checkpoint_dir}")

    candidates.sort(reverse=True)
    iteration, name = candidates[0]
    path = os.path.join(checkpoint_dir, name)
    print(f"  Found latest checkpoint: {path} (iteration {iteration:,})")
    return path


def _resolve_checkpoint(checkpoint: str | None, exp_name: str) -> str:
    """Resolve user-supplied --checkpoint, falling back to the newest run."""
    if checkpoint:
        path = checkpoint if os.path.isabs(checkpoint) else os.path.abspath(checkpoint)
        if os.path.isdir(path):
            return _find_latest_checkpoint(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    project_root = Path(__file__).resolve().parent.parent
    exp_root = project_root / "checkpoints" / exp_name
    if not exp_root.exists():
        # Fallback to legacy ``checkpoints/mlp`` location (matches train.py default).
        legacy = project_root / "checkpoints" / "mlp"
        if legacy.exists():
            exp_root = legacy
        else:
            raise FileNotFoundError(
                f"No checkpoints under {exp_root} or {legacy}. "
                "Pass --checkpoint <path>."
            )

    run_dirs = sorted(
        (d for d in exp_root.iterdir() if d.is_dir() and d.name.startswith("checkpoints_")),
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No checkpoints_* run directories in {exp_root}")
    return _find_latest_checkpoint(str(run_dirs[0]))


def _load_checkpoint(path: str) -> dict[str, Any]:
    print(f"  Loading checkpoint: {path}")
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    iteration = checkpoint.get("iteration", 0)
    print(f"    iteration: {iteration:,}")

    params = checkpoint.get("params")
    if params is None:
        raise KeyError(f"Checkpoint missing 'params': {path}")

    # Flax stores trainable parameters under ``params``; tolerate other layouts.
    try:
        std = params["params"]["std"]
        print(f"    action_std (mean |·|): {float(jnp.mean(jnp.abs(std))):.4f}")
    except (KeyError, TypeError):
        pass
    return checkpoint


# ============================================================================
# Config overrides for visualization
# ============================================================================

def _disable_randomization(env_cfg) -> None:
    """Turn off noise / DR / random resets for a clean rollout."""
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False

    if hasattr(env_cfg, "domain_rand"):
        dr = env_cfg.domain_rand
        for field in (
            "randomize_friction",
            "randomize_mass",
            "randomize_com_offset",
            "randomize_motor_strength",
            "randomize_Kp_factor",
            "randomize_Kd_factor",
            "randomize_dof_init",
            "randomize_action_latency",
            "randomize_obs_latency",
            "randomize_torques_latency",
            "push_robots",
            "push_robots_FM",
        ):
            if hasattr(dr, field):
                setattr(dr, field, False)


def _apply_command_overrides(env_cfg, args) -> None:
    """Pin the velocity command for visualization."""
    if not hasattr(env_cfg, "commands"):
        return

    cmd = env_cfg.commands
    if hasattr(cmd, "heading_command"):
        cmd.heading_command = False
        print("  commands.heading_command = False (using direct ang_vel_yaw)")
    if hasattr(cmd, "resampling_time"):
        cmd.resampling_time = 1.0e6
        print("  commands.resampling_time disabled")
    if hasattr(cmd, "zero_cmd_probability"):
        cmd.zero_cmd_probability = 0.0

    if hasattr(cmd, "ranges"):
        if args.cmd_vel_x is not None:
            cmd.ranges.lin_vel_x = (args.cmd_vel_x, args.cmd_vel_x)
            print(f"  command lin_vel_x: {args.cmd_vel_x}")
        if args.cmd_vel_y is not None:
            cmd.ranges.lin_vel_y = (args.cmd_vel_y, args.cmd_vel_y)
            print(f"  command lin_vel_y: {args.cmd_vel_y}")
        if args.cmd_vel_yaw is not None:
            cmd.ranges.ang_vel_yaw = (args.cmd_vel_yaw, args.cmd_vel_yaw)
            print(f"  command ang_vel_yaw: {args.cmd_vel_yaw}")


# ============================================================================
# Inference helpers
# ============================================================================

def _actor_obs_from_state(env_state) -> jnp.ndarray:
    """``State.obs`` may be a tensor or ``{"state", "privileged"}`` dict."""
    obs = env_state.obs
    if isinstance(obs, dict):
        return obs["state"]
    return obs


def _critic_obs_from_state(env_state, fallback) -> jnp.ndarray:
    obs = env_state.obs
    if isinstance(obs, dict):
        return obs.get("privileged", fallback)
    return fallback


# ============================================================================
# ⭐ 新增：地形观测调试函数
# ============================================================================

def _debug_terrain_observations(env_state, env, step: int, print_details: bool = False) -> None:
    """
    打印地形观测信息，用于验证多地形功能是否正常工作。
    
    Args:
        env_state: 环境状态
        env: 环境实例
        step: 当前步数
        print_details: 是否打印详细的高度扫描值
    """
    obs = env_state.obs
    if not isinstance(obs, dict):
        return
    
    privileged_obs = obs.get("privileged")
    if privileged_obs is None:
        return
    
    # 获取观测维度信息
    obs_size = env.observation_size
    actor_dim = obs_size.get("state", 48)

    # height_scan 的位置（在 VELOCITY_PRIVILEGED_SCHEMA 中定义），避免硬编码维度
    hs_field = next((f for f in VELOCITY_PRIVILEGED_SCHEMA if f.name == "height_scan"), None)
    if hs_field is None:
        # fallback：foot_contact_forces(60-71) 之后接 height_scan
        height_scan_start, height_scan_end = 72, 259
    else:
        height_scan_start, height_scan_end = hs_field.start, hs_field.stop
    privileged_dim = obs_size.get("privileged", height_scan_end)

    # 检查维度是否足够
    if privileged_obs.shape[-1] < height_scan_end:
        print(f"  ⚠️ Privileged obs dim {privileged_obs.shape[-1]} < {height_scan_end}, height_scan not available")
        return

    # 提取 height_scan
    height_scan = privileged_obs[..., height_scan_start:height_scan_end]
    
    # 计算统计数据
    height_scan_np = np.array(height_scan[0]) if height_scan.shape[0] > 0 else np.array(height_scan)
    is_all_zero = np.all(np.abs(height_scan_np) < 1e-6)
    
    # 打印信息
    print(f"\n[Terrain Debug] Step {step}:")
    print(f"  Privileged obs shape: {privileged_obs.shape}")
    print(f"  Height scan shape: {height_scan.shape}")
    print(f"  Height scan all zero? {is_all_zero}")
    
    if is_all_zero:
        print("  ⚠️ WARNING: height_scan is all zeros! Terrain query not working.")
        print("  Check:")
        print("    1. config.terrain.mesh_type != 'plane'")
        print("    2. env._terrain_height_field is not None")
        print("    3. 'terrain_query' in info dict")
    else:
        print(f"  Height scan range: [{height_scan_np.min():.4f}, {height_scan_np.max():.4f}]")
        print(f"  Height scan mean: {height_scan_np.mean():.4f}")
        
        if print_details:
            # 187 个点全打印太多，reshape 为 11×17 网格后隔点抽样打印 6 行 × 9 列
            scan_grid = height_scan_np.reshape(HEIGHT_SCAN_NUM_Y, HEIGHT_SCAN_NUM_X)
            sub_grid = scan_grid[::2, ::2]
            print(f"  Height scan values (subsampled {sub_grid.shape[0]}x{sub_grid.shape[1]} of "
                  f"{HEIGHT_SCAN_NUM_Y}x{HEIGHT_SCAN_NUM_X} grid):")
            for i in range(sub_grid.shape[0]):
                row_values = sub_grid[i]
                print(f"    Row {i}: [{', '.join([f'{v:.4f}' for v in row_values])}]")
    
    # 检查地形查询函数是否在环境中
    if hasattr(env, '_terrain_height_field'):
        if env._terrain_height_field is not None:
            print(f"  Terrain height field shape: {env._terrain_height_field.shape}")
            # 采样几个点验证
            if hasattr(env, '_query_terrain_height'):
                h_center = env._query_terrain_height(0.0, 0.0)
                h_pos = env._query_terrain_height(2.0, 2.0)
                print(f"  Query test: h(0,0)={h_center:.4f}, h(2,2)={h_pos:.4f}")
            else:
                print("  ⚠️ env._query_terrain_height not found")
        else:
            print("  ⚠️ env._terrain_height_field is None (plane terrain?)")
    else:
        print("  ⚠️ env has no _terrain_height_field attribute")


# ============================================================================
# Main
# ============================================================================

def play(args: argparse.Namespace) -> list[float]:
    print("=" * 80)
    print(f"MJX Play - Experiment: {args.exp}")
    print("=" * 80)
    print(f"  JAX backend: {jax.default_backend()} "
          f"(JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '<unset>')})")

    # ---- Build experiment + configs (mirrors train.py) ----------------------
    spec = get_experiment(args.exp)
    env_cfg = spec.build_env_config()
    ppo_cfg = spec.build_ppo_config()

    print("\n[Config] Disabling noise / domain randomization for visualization...")
    _disable_randomization(env_cfg)
    _apply_command_overrides(env_cfg, args)

    # Force a small env count for inference.
    num_envs = max(args.num_envs, args.num_render_envs)
    env_cfg.env.num_envs = num_envs

    # ---- Build environment directly (bypass build_env, which rebuilds cfg) --
    print(f"\n[1/4] Building environment ({num_envs} env(s))...")
    env_cls = load_symbol(spec.env_class_path)
    env = env_cls(config=env_cfg, component_name=spec.component_name)
    env.experiment_spec = spec
    obs_size = env.observation_size.get("state", 48)
    critic_obs_size = env.observation_size.get("privileged", 72)
    action_size = env.action_size
    print(f"  actor_obs: {obs_size}  critic_obs: {critic_obs_size}  action: {action_size}")

    # ⭐ 打印地形配置信息
    print(f"\n[Terrain] Configuration:")
    print(f"  mesh_type: {env_cfg.terrain.mesh_type}")
    if hasattr(env, '_terrain_height_field'):
        if env._terrain_height_field is not None:
            print(f"  height_field shape: {env._terrain_height_field.shape}")
            print(f"  height_field range: [{env._terrain_height_field.min():.3f}, {env._terrain_height_field.max():.3f}]")
        else:
            print("  height_field: None (plane terrain)")
    else:
        print("  ⚠️ env has no _terrain_height_field (did you modify generic_env.py?)")

    # Default video fps mirrors the sim cadence: real-time playback when
    # render_every == 1, slowed down proportionally otherwise.  Only computed
    # if the user didn't pin --fps explicitly.
    if args.fps is None:
        args.fps = (1.0 / env.dt) / max(args.render_every, 1)
        print(f"  derived video fps: {args.fps:.2f} (env.dt={env.dt:.4f}, "
              f"render_every={args.render_every})")

    # ---- Build the network exactly as train.py does -------------------------
    print("\n[2/4] Creating ActorCriticMLP...")
    network = ActorCriticMLP(
        action_size=action_size,
        actor_hidden_dims=tuple(ppo_cfg.network.policy_hidden_layer_sizes),
        critic_hidden_dims=tuple(ppo_cfg.network.value_hidden_layer_sizes),
        init_noise_std=ppo_cfg.network.init_noise_std,
        activation=ppo_cfg.network.activation,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_actor_obs = jnp.zeros((1, obs_size))
    dummy_critic_obs = jnp.zeros((1, critic_obs_size))
    dummy_params = network.init(init_rng, dummy_actor_obs, dummy_critic_obs)

    # ---- Load checkpoint or use the random init -----------------------------
    if args.random_policy:
        print("\n[3/4] Using RANDOM policy (no checkpoint)...")
        params = dummy_params
    else:
        print("\n[3/4] Loading checkpoint...")
        checkpoint_path = _resolve_checkpoint(args.checkpoint, args.exp)
        checkpoint = _load_checkpoint(checkpoint_path)
        params = checkpoint["params"]

    # ---- JIT-compile network + env -----------------------------------------
    print("\n[4/4] JIT compiling...")

    @jax.jit
    def policy_mean(p, obs):
        return network.apply(p, obs, method=network.act)

    @jax.jit
    def policy_stochastic(p, obs, critic_obs):
        return network.apply(p, obs, critic_obs)

    @jax.jit
    def update_episode_stats(total_rewards, episode_lengths, reward, done):
        """Vectorized per-step accumulator + reset-on-done.

        Returns (rewards_pre_reset, lengths_pre_reset, total_after_reset,
        lengths_after_reset, done_mask).  The pre-reset values are what we want
        to log when an episode finishes; the post-reset values feed the next
        step.  Using ``jnp.where`` keeps shapes static → no XLA recompiles.
        """
        new_total = total_rewards + reward
        new_lengths = episode_lengths + 1
        done_mask = done > 0.5
        next_total = jnp.where(done_mask, 0.0, new_total)
        next_lengths = jnp.where(done_mask, 0.0, new_lengths)
        return new_total, new_lengths, next_total, next_lengths, done_mask

    # Warm up so compile errors surface before the rollout window.
    _ = policy_mean(params, dummy_actor_obs)
    if not args.deterministic:
        _ = policy_stochastic(params, dummy_actor_obs, dummy_critic_obs)
    print("  ✓ network compiled")

    jit_reset = jax.jit(jax.vmap(env.reset))
    # Play path uses action clipping matching the original project.
    step_clipped = lambda state, action, odom: env.step(state, action, odom, clip_action=True)
    jit_step = jax.jit(jax.vmap(step_clipped, in_axes=(0, 0, 0)))

    # ⭐ 打印地形调试信息（重置后）
    rng, env_rng = jax.random.split(rng)
    env_rngs = jax.random.split(env_rng, num_envs)
    env_state = jit_reset(env_rngs)
    print("\n[Terrain] After reset:")
    _debug_terrain_observations(env_state, env, 0, print_details=True)

    # ---- Viewer setup -------------------------------------------------------
    if not args.save_video:
        print(f"\n[Viewer] Launching real-time viewer ({args.num_render_envs} robot(s))...")
        env.enable_viewer(True, num_render_envs=args.num_render_envs)

    # ---- Rollout ------------------------------------------------------------
    # 重新初始化状态（因为 viewer 可能已经修改了 env_state）
    rng, env_rng = jax.random.split(rng)
    env_rngs = jax.random.split(env_rng, num_envs)
    env_state = jit_reset(env_rngs)

    dummy_odom = jnp.zeros((num_envs, _ODOM_DIM))
    total_rewards = jnp.zeros(num_envs)
    episode_lengths = jnp.zeros(num_envs)
    episode_rewards: list[float] = []
    episode_count = 0
    frames: list[np.ndarray] = []

    print("\n" + "-" * 60)
    print(f"Running {args.episode_length} steps (Ctrl+C to stop)")
    print("-" * 60)
    start_time = time.perf_counter()
    step = 0

    # ⭐ 地形调试打印间隔
    terrain_debug_interval = 500  # 每500步打印一次

    try:
        for step in range(args.episode_length):
            actor_obs = _actor_obs_from_state(env_state)

            if args.deterministic:
                action = policy_mean(params, actor_obs)
            else:
                critic_obs = _critic_obs_from_state(env_state, actor_obs)
                action_mean, action_std, _ = policy_stochastic(params, actor_obs, critic_obs)
                rng, action_rng = jax.random.split(rng)
                action = action_mean + action_std * jax.random.normal(action_rng, action_mean.shape)
            # Match training: only NaN-guard. The env clips to action_clip itself.
            action = jnp.where(jnp.isfinite(action), action, 0.0)

            env_state = jit_step(env_state, action, dummy_odom)

            # Device-side accumulation + reset (no shape-dependent recompiles).
            (rewards_pre, lengths_pre,
             total_rewards, episode_lengths,
             done_mask) = update_episode_stats(
                total_rewards, episode_lengths, env_state.reward, env_state.done,
            )

            dones_np = np.asarray(done_mask)
            if dones_np.any():
                rewards_np = np.asarray(rewards_pre)
                lengths_np = np.asarray(lengths_pre)
                for idx in np.where(dones_np)[0]:
                    episode_rewards.append(float(rewards_np[idx]))
                    episode_count += 1
                    if episode_count <= 5 or episode_count % 10 == 0:
                        print(f"  Episode {episode_count}: reward={rewards_np[idx]:.2f}, "
                              f"length={lengths_np[idx]:.0f}")

            # ---- Rendering ----
            if args.save_video and step % args.render_every == 0:
                frames.append(env.render_frame(
                    env_state,
                    width=args.width,
                    height=args.height,
                    camera_mode="tracking",
                    track_body_id=1,
                    distance=1.5,
                    elevation=-20.0,
                    azimuth=135.0,
                ))
            elif not args.save_video:
                env.render(states=env_state, sync_frame_time=True)

            # ---- ⭐ 地形调试打印 ----
            if step % terrain_debug_interval == 0 and step > 0:
                _debug_terrain_observations(env_state, env, step, print_details=False)

            if (step + 1) % 100 == 0:
                elapsed = time.perf_counter() - start_time
                fps = (step + 1) / max(elapsed, 1e-6)
                base_quat = env_state.data.qpos[0, 3:7]
                world_lin_vel = np.array(env_state.data.qvel[0, 0:3])
                body_lin_vel = np.array(
                    math.rotate(env_state.data.qvel[0, 0:3], math.quat_inv(base_quat))
                )
                body_ang_vel = np.array(
                    math.rotate(env_state.data.qvel[0, 3:6], math.quat_inv(base_quat))
                )
                cmd = np.array(env_state.info.command[0])
                base_pos = np.array(env_state.data.qpos[0, 0:3])
                print(f"  Step {step + 1}/{args.episode_length} ({fps:.1f} FPS)")
                print(f"    cmd:    [vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, yaw={cmd[2]:.2f}]")
                print(f"    actual(body):  [vx={body_lin_vel[0]:.2f}, vy={body_lin_vel[1]:.2f}, "
                      f"vz={body_lin_vel[2]:.2f}, yaw={body_ang_vel[2]:.2f}]")
                print(f"    actual(world): [vx={world_lin_vel[0]:.2f}, vy={world_lin_vel[1]:.2f}, "
                      f"vz={world_lin_vel[2]:.2f}]")
                print(f"    pos:    [x={base_pos[0]:.2f}, y={base_pos[1]:.2f}, "
                      f"z={base_pos[2]:.2f}]")

    except KeyboardInterrupt:
        print("\nRollout interrupted by user.")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total steps: {step + 1}")
    print(f"Total time:  {elapsed:.1f}s")
    print(f"Average FPS: {(step + 1) / max(elapsed, 1e-6):.1f}")
    print(f"Episodes:    {episode_count}")
    if episode_rewards:
        print(f"Reward mean/min/max: {np.mean(episode_rewards):.2f} / "
              f"{np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")

    if args.save_video and frames:
        _save_frames(frames, args)

    if hasattr(env, "_viewer") and env._viewer is not None:
        env.close_viewer()

    return episode_rewards


# ============================================================================
# Video saving helper
# ============================================================================

def _save_frames(frames: list[np.ndarray], args) -> None:
    """Write the collected frames to disk.  Avoids the ``np.array(frames)``
    copy (which briefly doubles peak memory) by passing the list straight to
    the writer — both mediapy and imageio accept iterables of frames."""
    video_path = args.video_path or "play_output.mp4"
    print(f"\nSaving video to {video_path}...")
    duration = len(frames) / max(args.fps, 1e-6)

    if HAS_MEDIAPY:
        try:
            media.write_video(video_path, frames, fps=args.fps)
            print(f"  saved: {video_path} ({len(frames)} frames, {duration:.1f}s)")
            return
        except Exception as e:
            print(f"  mediapy failed ({e}), trying imageio fallback...")

    try:
        import imageio
        if video_path.endswith(".mp4"):
            gif_path = video_path.replace(".mp4", ".gif")
            imageio.mimsave(gif_path, frames, fps=args.fps)
            print(f"  saved (gif fallback): {gif_path} ({len(frames)} frames)")
        else:
            imageio.mimsave(video_path, frames, fps=args.fps)
            print(f"  saved: {video_path} ({len(frames)} frames)")
        return
    except ImportError:
        pass

    npy_path = (args.video_path or "play_output.mp4").replace(".mp4", "_frames.npy")
    np.save(npy_path, np.asarray(frames))
    print(f"  saved frames as numpy: {npy_path} ({len(frames)} frames)")
    print("  install ffmpeg + mediapy/imageio to write video files")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a trained MJX velocity policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment + checkpoint (mirrors train.py).
    parser.add_argument("--exp", default="xg_velocity", choices=list_experiments(),
                        help="Experiment registered in cfg/experiments/")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to iteration_*.pkl (or its dir). Defaults to latest run.")
    parser.add_argument("--random_policy", action="store_true",
                        help="Use the random init instead of loading a checkpoint")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "gpu", "tpu"),
                        help="JAX backend. Default auto-detect (GPU if available).")

    # Rollout.
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use action_mean (no Gaussian sampling). Default True.")

    # Command overrides.
    parser.add_argument("--cmd_vel_x", type=float, default=1.0)
    parser.add_argument("--cmd_vel_y", type=float, default=0.0)
    parser.add_argument("--cmd_vel_yaw", type=float, default=0.0)

    # Rendering.
    parser.add_argument("--num_render_envs", type=int, default=1,
                        help="Robots in the multi-robot grid")
    parser.add_argument("--render_every", type=int, default=2,
                        help="Render every N steps when saving video")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=None,
                        help="Video fps. Defaults to (1/env.dt)/render_every "
                             "so playback runs at real-time.")

    # Video output.
    parser.add_argument("--save_video", action="store_true",
                        help="Render offscreen to a video file instead of the viewer")
    parser.add_argument("--video_path", type=str, default=None)

    args = parser.parse_args()

    if args.num_envs < args.num_render_envs:
        args.num_envs = args.num_render_envs
        print(f"Note: num_envs raised to {args.num_envs} to match num_render_envs")

    play(args)


if __name__ == "__main__":
    main()