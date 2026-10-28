"""ROS2 (Jazzy) play script for the MJX velocity experiments.

Subscribes to ``cmd_vel`` (geometry_msgs/Twist) and drives a trained policy.
Checkpoint selection mirrors play.py.

Usage:
    # source your ROS2 workspace first
    source /opt/ros/jazzy/setup.bash

    python play_ros2.py --checkpoint checkpoints/xg_velocity/.../iteration_5000.pkl
    python play_ros2.py                     # latest checkpoint of xg_velocity
    python play_ros2.py --random_policy     # smoke test

    # publish commands from another terminal
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def _configure_jax_device() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")


def _pre_parse_device() -> None:
    for i, tok in enumerate(sys.argv[1:], start=1):
        if tok == "--device" and i < len(sys.argv) - 1:
            os.environ["JAX_PLATFORMS"] = sys.argv[i + 1]
            return
        if tok.startswith("--device="):
            os.environ["JAX_PLATFORMS"] = tok.split("=", 1)[1]
            return


_pre_parse_device()
_configure_jax_device()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from brax import math

from cfg.experiments import get_experiment, list_experiments
from envs import load_symbol
from networks import ActorCriticMLP


_ODOM_DIM = 3


# ============================================================================
# ROS2 bridge
# ============================================================================

class CmdVelBridge:
    """Runs a ROS2 node in a background thread, exposes the latest cmd_vel.

    Thread-safety: the callback only writes to ``self._cmd`` under a lock;
    the main sim loop reads a snapshot via ``get()``.
    """

    def __init__(self, topic: str = "cmd_vel", timeout_s: float = 0.5):
        # Import here so the rest of the script can be imported without ROS.
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist

        self._rclpy = rclpy
        self._lock = threading.Lock()
        # (vx, vy, yaw_rate)
        self._cmd = np.zeros(3, dtype=np.float32)
        self._last_msg_time: float | None = None
        self._timeout_s = timeout_s
        self._shutdown = threading.Event()

        rclpy.init(args=None)
        self._node = Node("mjx_play_cmd_vel")

        self._sub = self._node.create_subscription(
            Twist, topic, self._on_cmd_vel, 10,
        )
        # Watchdog: zero the command if no message arrives for timeout_s.
        self._node.create_timer(0.1, self._watchdog)

        self._node.get_logger().info(
            f"Subscribed to '{topic}' (geometry_msgs/Twist). "
            f"Command zeroes after {timeout_s:.1f}s of silence."
        )

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _on_cmd_vel(self, msg) -> None:
        with self._lock:
            # Twist.linear.x/y = body-frame forward/left (m/s)
            # Twist.angular.z    = yaw rate (rad/s)
            self._cmd[0] = float(msg.linear.x)
            self._cmd[1] = float(msg.linear.y)
            self._cmd[2] = float(msg.angular.z)
            self._last_msg_time = time.monotonic()

    def _watchdog(self) -> None:
        with self._lock:
            if self._last_msg_time is None:
                return
            if time.monotonic() - self._last_msg_time > self._timeout_s:
                self._cmd[:] = 0.0

    def _spin(self) -> None:
        try:
            while not self._shutdown.is_set():
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
        except Exception as e:  # pragma: no cover
            print(f"[CmdVelBridge] spin loop exited: {e}")

    def get(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()

    def close(self) -> None:
        self._shutdown.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


# ============================================================================
# Checkpoint helpers (same as play.py)
# ============================================================================

def _find_latest_checkpoint(checkpoint_dir: str) -> str:
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

    try:
        std = params["params"]["std"]
        print(f"    action_std (mean |·|): {float(jnp.mean(jnp.abs(std))):.4f}")
    except (KeyError, TypeError):
        pass
    return checkpoint


# ============================================================================
# Config overrides
# ============================================================================

def _disable_randomization(env_cfg) -> None:
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False

    if hasattr(env_cfg, "domain_rand"):
        dr = env_cfg.domain_rand
        for field in (
            "randomize_friction", "randomize_mass", "randomize_com_offset",
            "randomize_motor_strength", "randomize_Kp_factor", "randomize_Kd_factor",
            "randomize_dof_init", "randomize_action_latency", "randomize_obs_latency",
            "randomize_torques_latency", "push_robots", "push_robots_FM",
        ):
            if hasattr(dr, field):
                setattr(dr, field, False)


def _freeze_command_sampling(env_cfg) -> None:
    """Make the env sample a *fixed* placeholder command.

    We'll overwrite ``state.info.command`` every step with the ROS cmd_vel, so
    the sampled value is irrelevant.  We just need resampling to stop.
    """
    if not hasattr(env_cfg, "commands"):
        return
    cmd = env_cfg.commands
    if hasattr(cmd, "heading_command"):
        cmd.heading_command = False
    if hasattr(cmd, "resampling_time"):
        cmd.resampling_time = 1.0e6
    if hasattr(cmd, "zero_cmd_probability"):
        cmd.zero_cmd_probability = 0.0


# ============================================================================
# Inference helpers
# ============================================================================

def _actor_obs_from_state(env_state) -> jnp.ndarray:
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
# Main
# ============================================================================

def play(args: argparse.Namespace) -> None:
    print("=" * 80)
    print(f"MJX ROS2 Play - Experiment: {args.exp}")
    print("=" * 80)
    print(f"  JAX backend: {jax.default_backend()} "
          f"(JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '<unset>')})")

    # ---- Experiment config --------------------------------------------------
    spec = get_experiment(args.exp)
    env_cfg = spec.build_env_config()
    ppo_cfg = spec.build_ppo_config()

    _disable_randomization(env_cfg)
    _freeze_command_sampling(env_cfg)

    num_envs = max(args.num_envs, args.num_render_envs)
    env_cfg.env.num_envs = num_envs

    # ---- Build env ----------------------------------------------------------
    print(f"\n[1/5] Building environment ({num_envs} env(s))...")
    env_cls = load_symbol(spec.env_class_path)
    env = env_cls(config=env_cfg, component_name=spec.component_name)
    env.experiment_spec = spec
    obs_size = env.observation_size.get("state", 48)
    critic_obs_size = env.observation_size.get("privileged", 72)
    action_size = env.action_size
    print(f"  actor_obs: {obs_size}  critic_obs: {critic_obs_size}  action: {action_size}")

    # ---- Network ------------------------------------------------------------
    print("\n[2/5] Creating ActorCriticMLP...")
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

    # ---- Checkpoint ---------------------------------------------------------
    if args.random_policy:
        print("\n[3/5] Using RANDOM policy (no checkpoint)...")
        params = dummy_params
    else:
        print("\n[3/5] Loading checkpoint...")
        checkpoint_path = _resolve_checkpoint(args.checkpoint, args.exp)
        checkpoint = _load_checkpoint(checkpoint_path)
        params = checkpoint["params"]

    # ---- JIT ----------------------------------------------------------------
    print("\n[4/5] JIT compiling...")

    @jax.jit
    def policy_mean(p, obs):
        return network.apply(p, obs, method=network.act)

    @jax.jit
    def policy_stochastic(p, obs, critic_obs):
        return network.apply(p, obs, critic_obs)

    jit_reset = jax.jit(jax.vmap(env.reset))
    step_clipped = lambda state, action, odom: env.step(state, action, odom, clip_action=True)
    jit_step = jax.jit(jax.vmap(step_clipped, in_axes=(0, 0, 0)))

    _ = policy_mean(params, dummy_actor_obs)
    if not args.deterministic:
        _ = policy_stochastic(params, dummy_actor_obs, dummy_critic_obs)
    print("  ✓ network compiled")

    # ---- ROS2 ---------------------------------------------------------------
    print(f"\n[5/5] Starting ROS2 cmd_vel bridge (topic='{args.cmd_vel_topic}')...")
    bridge = CmdVelBridge(topic=args.cmd_vel_topic, timeout_s=args.cmd_vel_timeout)

    # ---- Viewer -------------------------------------------------------------
    if not args.save_video:
        print(f"\n[Viewer] Launching real-time viewer ({args.num_render_envs} robot(s))...")
        env.enable_viewer(True, num_render_envs=args.num_render_envs)

    # ---- Reset --------------------------------------------------------------
    rng, env_rng = jax.random.split(rng)
    env_rngs = jax.random.split(env_rng, num_envs)
    env_state = jit_reset(env_rngs)

    dummy_odom = jnp.zeros((num_envs, _ODOM_DIM))

    print("\n" + "-" * 60)
    print(f"Running {args.episode_length} steps (Ctrl+C to stop)")
    print("Publish to /cmd_vel to drive.  Zero after "
          f"{args.cmd_vel_timeout:.1f}s of silence.")
    print("-" * 60)

    start_time = time.perf_counter()
    step = 0

    def _inject_command(state, cmd_np: np.ndarray):
        """Overwrite ``state.info.command`` with the ROS cmd_vel.

        ``cmd_np`` is (3,) → broadcast to (num_envs, 3).
        Env reads ``state.info.command`` in its reward.  If your env stores the
        command elsewhere, adjust this helper accordingly.
        """
        cmd = jnp.asarray(cmd_np, dtype=state.info.command.dtype)
        cmd = jnp.broadcast_to(cmd, state.info.command.shape)
        return state.replace(info=state.info.replace(command=cmd))

    try:
        for step in range(args.episode_length):
            # 1) Read latest ROS command and inject it.
            cmd = bridge.get()
            env_state = _inject_command(env_state, cmd)

            # 2) Policy forward.
            actor_obs = _actor_obs_from_state(env_state)
            if args.deterministic:
                action = policy_mean(params, actor_obs)
            else:
                critic_obs = _critic_obs_from_state(env_state, actor_obs)
                action_mean, action_std, _ = policy_stochastic(params, actor_obs, critic_obs)
                rng, action_rng = jax.random.split(rng)
                action = action_mean + action_std * jax.random.normal(action_rng, action_mean.shape)
            action = jnp.where(jnp.isfinite(action), action, 0.0)

            # 3) Step.
            env_state = jit_step(env_state, action, dummy_odom)

            # 4) Auto-reset on done so the robot keeps going.
            done_np = np.asarray(env_state.done) > 0.5
            if done_np.any():
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, num_envs)
                fresh = jit_reset(reset_rngs)

                done_mask = jnp.asarray(done_np)  # (num_envs,)

                def _select(new, old):
                    if not hasattr(old, "ndim") or old.ndim == 0:
                        return old
                    m = done_mask.reshape((num_envs,) + (1,) * (old.ndim - 1))
                    return jnp.where(m, new, old)

                # 1) Replace data wholesale (reset and step produce the same data layout).
                new_data = jax.tree.map(_select, fresh.data, env_state.data)

                # 2) For info, only carry over the few keys we actually use; leave
                #    everything else as-is so step() refills it next iteration.
                info_updates = {}
                for key in ("command", "step", "commands", "episode_length"):
                    if hasattr(fresh.info, key) or (isinstance(fresh.info, dict) and key in fresh.info):
                        old_val = (env_state.info[key] if isinstance(env_state.info, dict)
                                else getattr(env_state.info, key))
                        new_val = (fresh.info[key] if isinstance(fresh.info, dict)
                                else getattr(fresh.info, key))
                        info_updates[key] = _select(new_val, old_val)

                if isinstance(env_state.info, dict):
                    new_info = dict(env_state.info)
                    new_info.update(info_updates)
                    env_state = env_state.replace(data=new_data, info=new_info)
                else:
                    env_state = env_state.replace(data=new_data, info=env_state.info.replace(**info_updates))

                # 3) Re-inject the current ROS command (reset overwrote it).
                env_state = _inject_command(env_state, bridge.get())

            # 5) Render.
            if not args.save_video:
                env.render(states=env_state, sync_frame_time=True)

            # 6) Logging.
            if (step + 1) % 100 == 0:
                elapsed = time.perf_counter() - start_time
                fps = (step + 1) / max(elapsed, 1e-6)
                base_quat = env_state.data.qpos[0, 3:7]
                body_lin_vel = np.array(
                    math.rotate(env_state.data.qvel[0, 0:3], math.quat_inv(base_quat))
                )
                body_ang_vel = np.array(
                    math.rotate(env_state.data.qvel[0, 3:6], math.quat_inv(base_quat))
                )
                cur_cmd = np.array(env_state.info.command[0])
                base_pos = np.array(env_state.data.qpos[0, 0:3])
                print(f"  Step {step + 1}/{args.episode_length} ({fps:.1f} FPS)")
                print(f"    cmd (ros):    [vx={cur_cmd[0]:.2f}, vy={cur_cmd[1]:.2f}, "
                      f"yaw={cur_cmd[2]:.2f}]")
                print(f"    actual(body): [vx={body_lin_vel[0]:.2f}, vy={body_lin_vel[1]:.2f}, "
                      f"vz={body_lin_vel[2]:.2f}, yaw={body_ang_vel[2]:.2f}]")
                print(f"    pos:          [x={base_pos[0]:.2f}, y={base_pos[1]:.2f}, "
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

    bridge.close()
    if hasattr(env, "_viewer") and env._viewer is not None:
        env.close_viewer()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROS2 cmd_vel-driven play script for MJX velocity policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--exp", default="xg_velocity", choices=list_experiments())
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to iteration_*.pkl (or its dir). Defaults to latest run.")
    parser.add_argument("--random_policy", action="store_true")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "gpu", "tpu"))

    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=200000,
                        help="Max sim steps before exiting (default: run ~forever)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=True)

    parser.add_argument("--num_render_envs", type=int, default=1)

    # ROS2
    parser.add_argument("--cmd_vel_topic", type=str, default="cmd_vel",
                        help="Twist topic name (default: cmd_vel)")
    parser.add_argument("--cmd_vel_timeout", type=float, default=0.5,
                        help="Seconds without a message before zeroing the command")

    # Kept for CLI compat, not used in ROS path.
    parser.add_argument("--save_video", action="store_true",
                        help="(unsupported in ROS2 play; viewer only)")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--render_every", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=None)

    args = parser.parse_args()

    if args.save_video:
        print("Note: --save_video is not implemented in play_ros2.py; "
              "running the live viewer instead.")

    if args.num_envs < args.num_render_envs:
        args.num_envs = args.num_render_envs

    play(args)


if __name__ == "__main__":
    main()