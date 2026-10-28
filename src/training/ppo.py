# from __future__ import annotations

# import os
# import pickle
# import statistics
# import time
# from collections import deque
# from typing import Any

# import jax
# import jax.numpy as jnp
# import numpy as np
# import optax
# from flax import struct
# from torch.utils.tensorboard import SummaryWriter


# @struct.dataclass
# class TrainingState:
#     params: Any
#     opt_state: Any
#     step: int
#     rng: jax.Array


# @struct.dataclass
# class RolloutBuffer:
#     obs: jax.Array
#     critic_obs: jax.Array
#     actions: jax.Array
#     values: jax.Array
#     log_probs: jax.Array
#     mu: jax.Array
#     sigma: jax.Array
#     advantages: jax.Array
#     returns: jax.Array


# def sample_action(rng, mean, std):
#     """Sample an action from a diagonal Gaussian policy."""
#     return mean + std * jax.random.normal(rng, mean.shape)


# def log_prob_normal(x, mean, std):
#     """Elementwise log probability of a normal distribution."""
#     var = std**2
#     log_scale = jnp.log(std)
#     return -((x - mean) ** 2) / (2 * var) - log_scale - 0.5 * jnp.log(2 * jnp.pi)


# def compute_gae(rewards, values, dones, gamma, lam):
#     """Compute Generalized Advantage Estimation."""
#     advantages = []
#     gae = jnp.zeros_like(values[0])

#     for t in reversed(range(len(rewards))):
#         next_value = jnp.zeros_like(values[t]) if t == len(rewards) - 1 else values[t + 1]
#         delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
#         gae = delta + gamma * lam * (1 - dones[t]) * gae
#         advantages.insert(0, gae)

#     advantages = jnp.stack(advantages)
#     returns = advantages + jnp.stack(values)
#     return advantages, returns


# def ppo_loss(
#     params,
#     network,
#     obs,
#     critic_obs,
#     actions,
#     old_log_probs,
#     old_values,
#     old_mu,
#     old_sigma,
#     advantages,
#     returns,
#     clip_param,
#     vf_coef,
#     entropy_coef,
# ):
#     """PPO clipped objective with asymmetric critic observations."""
#     action_mean, action_std, values = network.apply(params, obs, critic_obs)

#     log_probs = jnp.sum(log_prob_normal(actions, action_mean, action_std), axis=-1)
#     ratio = jnp.exp(log_probs - old_log_probs)
#     surr1 = ratio * advantages
#     surr2 = jnp.clip(ratio, 1 - clip_param, 1 + clip_param) * advantages
#     policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

#     value_pred_clipped = old_values + jnp.clip(values - old_values, -clip_param, clip_param)
#     value_loss1 = jnp.square(values - returns)
#     value_loss2 = jnp.square(value_pred_clipped - returns)
#     value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))

#     entropy = 0.5 * (1 + jnp.log(2 * jnp.pi)) + jnp.log(action_std)
#     entropy = jnp.mean(jnp.sum(entropy, axis=-1))
#     loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

#     kl = jnp.sum(
#         jnp.log(action_std / old_sigma)
#         + (old_sigma**2 + (old_mu - action_mean) ** 2) / (2 * action_std**2)
#         - 0.5,
#         axis=-1,
#     )

#     metrics = {
#         "total_loss": loss,
#         "policy_loss": policy_loss,
#         "value_loss": value_loss,
#         "entropy": entropy,
#         "mean_kl": jnp.mean(kl),
#         "mean_std": jnp.mean(action_std),
#     }
#     return loss, metrics


# def save_checkpoint(path, state, iteration, reward_stats=None):
#     """Save trainer checkpoint."""
#     checkpoint = {
#         "params": state.params,
#         "opt_state": state.opt_state,
#         "step": state.step,
#         "rng": np.array(state.rng),
#         "iteration": iteration,
#     }
#     if reward_stats:
#         checkpoint["reward_stats"] = reward_stats

#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "wb") as f:
#         pickle.dump(checkpoint, f)
#     print(f"  Saved checkpoint: {path}")


# def load_checkpoint(path):
#     """Load trainer checkpoint."""
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def _flatten_rollout(xs):
#     stacked = jnp.stack(xs, axis=0)
#     if stacked.ndim == 3:
#         return stacked.reshape(-1, stacked.shape[-1])
#     return stacked.reshape(-1)


# def train(
#     env,
#     network,
#     num_envs: int = 4096,
#     num_steps_per_env: int = 24,
#     num_learning_epochs: int = 5,
#     num_mini_batches: int = 4,
#     learning_rate: float = 1e-3,
#     gamma: float = 0.99,
#     lam: float = 0.95,
#     clip_param: float = 0.2,
#     vf_coef: float = 1.0,
#     entropy_coef: float = 0.01,
#     max_grad_norm: float = 1.0,
#     max_iterations: int = 10000,
#     desired_kl: float = 0.01,
#     seed: int = 42,
#     log_dir: str = None,
#     checkpoint_dir: str = None,
#     save_interval: int = 50,
#     resume_checkpoint: str = None,
# ):
#     """Train an MJX environment with PPO."""
#     print("\n" + "=" * 80)
#     print("Pure MLP PPO Training")
#     print("=" * 80)
#     print(f"  num_envs: {num_envs}")
#     print(f"  num_steps_per_env: {num_steps_per_env}")
#     print(f"  num_learning_epochs: {num_learning_epochs}")
#     print(f"  num_mini_batches: {num_mini_batches}")
#     print(f"  learning_rate: {learning_rate}")
#     print(f"  gamma: {gamma}")
#     print(f"  lam: {lam}")
#     print(f"  clip_param: {clip_param}")
#     print(f"  vf_coef: {vf_coef}")
#     print(f"  entropy_coef: {entropy_coef}")
#     print(f"  desired_kl: {desired_kl}")
#     print("=" * 80 + "\n")

#     if checkpoint_dir is None:
#         checkpoint_dir = (
#             log_dir.replace("tensorboard_", "checkpoints_") if log_dir else "checkpoints/mlp"
#         )

#     rng = jax.random.PRNGKey(seed)
#     obs_size = env.observation_size.get("state", 48)
#     critic_obs_size = env.observation_size.get("privileged", 72)
#     action_size = env.action_size
#     print(f"  Actor obs size: {obs_size}")
#     print(f"  Critic obs size: {critic_obs_size}")

#     rng, init_rng = jax.random.split(rng)
#     params = network.init(
#         init_rng,
#         jnp.zeros((1, obs_size)),
#         jnp.zeros((1, critic_obs_size)),
#     )

#     current_lr = learning_rate
#     min_lr = 1e-5
#     max_lr = 1e-3
#     optimizer = optax.chain(
#         optax.clip_by_global_norm(max_grad_norm),
#         optax.inject_hyperparams(optax.adam)(learning_rate=current_lr, eps=1e-5),
#     )
#     opt_state = optimizer.init(params)
#     state = TrainingState(params=params, opt_state=opt_state, step=0, rng=rng)

#     start_iteration = 0
#     if resume_checkpoint and os.path.exists(resume_checkpoint):
#         print(f"  Resuming from checkpoint: {resume_checkpoint}")
#         checkpoint = load_checkpoint(resume_checkpoint)
#         start_iteration = checkpoint["iteration"] + 1
#         state = TrainingState(
#             params=checkpoint["params"],
#             opt_state=checkpoint["opt_state"],
#             step=checkpoint["step"],
#             rng=jnp.array(checkpoint["rng"]),
#         )
#         print(f"  Resumed from iteration {start_iteration}")

#     writer = None
#     if log_dir:
#         os.makedirs(log_dir, exist_ok=True)
#         writer = SummaryWriter(log_dir)

#     print("  [1/3] Initializing environments...")
#     rng, env_rng = jax.random.split(rng)
#     env_rngs = jax.random.split(env_rng, num_envs)

#     print("  [2/3] JIT compiling env...")
#     jit_reset = jax.jit(jax.vmap(env.reset))
#     env_state = jit_reset(env_rngs)
#     jax.block_until_ready(env_state.obs["state"])

#     step_unclipped = lambda state, action, odom: env.step(state, action, odom, clip_action=False)
#     jit_step = jax.jit(jax.vmap(step_unclipped, in_axes=(0, 0, 0)))
#     dummy_action = jnp.zeros((num_envs, action_size))
#     dummy_odom = jnp.zeros((num_envs, 3))
#     compiled_state = jit_step(env_state, dummy_action, dummy_odom)
#     jax.block_until_ready(compiled_state)
#     print("        ✓ env compiled!")

#     print("  [3/3] JIT compiling network...")
#     jit_network = jax.jit(network.apply)
#     compiled_network = jit_network(
#         state.params,
#         jnp.zeros((num_envs, obs_size)),
#         jnp.zeros((num_envs, critic_obs_size)),
#     )
#     jax.block_until_ready(compiled_network)
#     print("        ✓ network compiled!")

#     @jax.jit
#     def train_step(params, opt_state, buffer):
#         def loss_fn(p):
#             return ppo_loss(
#                 p,
#                 network,
#                 buffer.obs,
#                 buffer.critic_obs,
#                 buffer.actions,
#                 buffer.log_probs,
#                 buffer.values,
#                 buffer.mu,
#                 buffer.sigma,
#                 buffer.advantages,
#                 buffer.returns,
#                 clip_param,
#                 vf_coef,
#                 entropy_coef,
#             )

#         (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
#         updates, new_opt_state = optimizer.update(grads, opt_state, params)
#         new_params = optax.apply_updates(params, updates)
#         return new_params, new_opt_state, metrics

#     print("  JIT compiling train_step...")
#     dummy_buffer = RolloutBuffer(
#         obs=jnp.zeros((1000, obs_size)),
#         critic_obs=jnp.zeros((1000, critic_obs_size)),
#         actions=jnp.zeros((1000, action_size)),
#         values=jnp.zeros(1000),
#         log_probs=jnp.zeros(1000),
#         mu=jnp.zeros((1000, action_size)),
#         sigma=jnp.ones((1000, action_size)),
#         advantages=jnp.zeros(1000),
#         returns=jnp.zeros(1000),
#     )
#     compiled_train = train_step(state.params, state.opt_state, dummy_buffer)
#     jax.block_until_ready(compiled_train)
#     print("        ✓ train_step compiled!")

#     rewbuffer = deque(maxlen=100)
#     lenbuffer = deque(maxlen=100)
#     cur_reward_sum = jnp.zeros(num_envs)
#     cur_episode_length = jnp.zeros(num_envs)

#     print("\n" + "=" * 80)
#     print("Starting training loop...")
#     print(f"  Training from iteration {start_iteration} to {start_iteration + max_iterations}")
#     print("=" * 80)
#     start_time = time.time()

#     for iteration in range(start_iteration, start_iteration + max_iterations):
#         iter_start = time.time()
#         obs_buffer = []
#         critic_obs_buffer = []
#         actions_buffer = []
#         rewards_buffer = []
#         dones_buffer = []
#         values_buffer = []
#         log_probs_buffer = []
#         mu_buffer = []
#         sigma_buffer = []
#         # Per-iter device-side dumps of completed-episode reward/length.  We
#         # write the episode total at the done step, and NaN elsewhere; the
#         # whole stack is sync'd to host ONCE at the end of the rollout (vs
#         # the old O(num_steps_per_env × done_envs) per-step sync).
#         ep_reward_dump = []
#         ep_length_dump = []

#         for _ in range(num_steps_per_env):
#             actor_obs = env_state.obs["state"]
#             critic_obs = env_state.obs.get("privileged", actor_obs)

#             action_mean, action_std, value = jit_network(state.params, actor_obs, critic_obs)

#             rng, action_rng = jax.random.split(state.rng)
#             state = state.replace(rng=rng)
#             action = sample_action(action_rng, action_mean, action_std)
#             action = jnp.where(jnp.isfinite(action), action, 0.0)
#             log_prob = jnp.sum(log_prob_normal(action, action_mean, action_std), axis=-1)

#             obs_buffer.append(actor_obs)
#             critic_obs_buffer.append(critic_obs)
#             actions_buffer.append(action)
#             values_buffer.append(value)
#             log_probs_buffer.append(log_prob)
#             mu_buffer.append(action_mean)
#             sigma_buffer.append(jnp.broadcast_to(action_std, action_mean.shape))

#             env_state = jit_step(env_state, action, jnp.zeros((num_envs, 3)))
#             rewards_buffer.append(env_state.reward)
#             dones_buffer.append(env_state.done)

#             cur_reward_sum = cur_reward_sum + env_state.reward
#             cur_episode_length = cur_episode_length + 1

#             # Device-side: stash episode totals where done, NaN elsewhere.  No
#             # host sync here — keeps the GPU pipeline full.
#             done_mask = env_state.done > 0.5
#             ep_reward_dump.append(
#                 jnp.where(done_mask, cur_reward_sum, jnp.nan)
#             )
#             ep_length_dump.append(
#                 jnp.where(done_mask, cur_episode_length, jnp.nan)
#             )
#             cur_reward_sum = jnp.where(done_mask, 0.0, cur_reward_sum)
#             cur_episode_length = jnp.where(done_mask, 0.0, cur_episode_length)

#         # Single batched sync per iter: pull (T, num_envs) reward/length tables
#         # to host, then push the finite (non-NaN) entries into the logging deques.
#         all_ep_rewards = np.asarray(jnp.stack(ep_reward_dump))
#         all_ep_lengths = np.asarray(jnp.stack(ep_length_dump))
#         finished_mask = ~np.isnan(all_ep_rewards)
#         if finished_mask.any():
#             rewbuffer.extend(all_ep_rewards[finished_mask].tolist())
#             lenbuffer.extend(all_ep_lengths[finished_mask].tolist())

#         advantages, returns = compute_gae(rewards_buffer, values_buffer, dones_buffer, gamma, lam)
#         buffer = RolloutBuffer(
#             obs=_flatten_rollout(obs_buffer),
#             critic_obs=_flatten_rollout(critic_obs_buffer),
#             actions=_flatten_rollout(actions_buffer),
#             values=_flatten_rollout(values_buffer),
#             log_probs=_flatten_rollout(log_probs_buffer),
#             mu=_flatten_rollout(mu_buffer),
#             sigma=_flatten_rollout(sigma_buffer),
#             advantages=advantages.reshape(-1),
#             returns=returns.reshape(-1),
#         )
#         buffer = buffer.replace(
#             advantages=(buffer.advantages - jnp.mean(buffer.advantages))
#             / (jnp.std(buffer.advantages) + 1e-8)
#         )

#         batch_size = buffer.obs.shape[0]
#         mini_batch_size = batch_size // num_mini_batches
#         total_metrics = {}

#         for _ in range(num_learning_epochs):
#             rng, perm_rng = jax.random.split(state.rng)
#             state = state.replace(rng=rng)
#             perm = jax.random.permutation(perm_rng, batch_size)

#             for mb in range(num_mini_batches):
#                 mb_indices = perm[mb * mini_batch_size : (mb + 1) * mini_batch_size]
#                 mb_buffer = RolloutBuffer(
#                     obs=buffer.obs[mb_indices],
#                     critic_obs=buffer.critic_obs[mb_indices],
#                     actions=buffer.actions[mb_indices],
#                     values=buffer.values[mb_indices],
#                     log_probs=buffer.log_probs[mb_indices],
#                     mu=buffer.mu[mb_indices],
#                     sigma=buffer.sigma[mb_indices],
#                     advantages=buffer.advantages[mb_indices],
#                     returns=buffer.returns[mb_indices],
#                 )

#                 new_params, new_opt_state, metrics = train_step(
#                     state.params, state.opt_state, mb_buffer
#                 )
#                 state = state.replace(
#                     params=new_params,
#                     opt_state=new_opt_state,
#                     step=state.step + 1,
#                 )

#                 for key, value in metrics.items():
#                     total_metrics.setdefault(key, []).append(float(value))

#         for key in total_metrics:
#             total_metrics[key] = statistics.mean(total_metrics[key])

#         mean_kl = total_metrics.get("mean_kl", 0.0)
#         old_lr = current_lr
#         if mean_kl > desired_kl * 2.0:
#             current_lr = max(current_lr / 1.5, min_lr)
#         elif mean_kl < desired_kl * 0.5:
#             current_lr = min(current_lr * 1.5, max_lr)

#         if current_lr != old_lr:
#             inner_opt_state = state.opt_state[1]
#             if hasattr(inner_opt_state, "hyperparams"):
#                 inner_opt_state.hyperparams["learning_rate"] = jnp.array(current_lr)

#         iter_time = time.time() - iter_start
#         if iteration % 10 == 0:
#             mean_reward = statistics.mean(rewbuffer) if rewbuffer else 0.0
#             mean_length = statistics.mean(lenbuffer) if lenbuffer else 0.0
#             mean_std = total_metrics.get("mean_std", 0.0)
#             print(
#                 f"Iter {iteration:5d} | reward: {mean_reward:7.2f} | "
#                 f"len: {mean_length:6.1f} | std: {mean_std:.3f} | "
#                 f"kl: {mean_kl:.4f} | lr: {current_lr:.2e} | time: {iter_time:.2f}s"
#             )

#             if writer:
#                 writer.add_scalar("train/mean_reward", mean_reward, iteration)
#                 writer.add_scalar("train/mean_episode_length", mean_length, iteration)
#                 writer.add_scalar("train/mean_std", mean_std, iteration)
#                 writer.add_scalar("train/mean_kl", mean_kl, iteration)
#                 writer.add_scalar("train/learning_rate", current_lr, iteration)
#                 for key, value in total_metrics.items():
#                     writer.add_scalar(f"loss/{key}", value, iteration)

#         if iteration % save_interval == 0 and iteration > start_iteration:
#             checkpoint_path = os.path.join(checkpoint_dir, f"iteration_{iteration}.pkl")
#             save_checkpoint(checkpoint_path, state, iteration)

#     checkpoint_path = os.path.join(
#         checkpoint_dir, f"iteration_{start_iteration + max_iterations}.pkl"
#     )
#     save_checkpoint(checkpoint_path, state, start_iteration + max_iterations)

#     total_time = time.time() - start_time
#     print(f"\nTraining completed in {total_time / 3600:.2f} hours")

#     if writer:
#         writer.close()

#     return state


# __all__ = [
#     "RolloutBuffer",
#     "TrainingState",
#     "compute_gae",
#     "load_checkpoint",
#     "log_prob_normal",
#     "ppo_loss",
#     "sample_action",
#     "save_checkpoint",
#     "train",
# ]
from __future__ import annotations

import os
import pickle
import statistics
import time
from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from torch.utils.tensorboard import SummaryWriter


@struct.dataclass
class TrainingState:
    params: Any
    opt_state: Any
    step: int
    rng: jax.Array


@struct.dataclass
class RolloutBuffer:
    obs: jax.Array
    critic_obs: jax.Array
    actions: jax.Array
    values: jax.Array
    log_probs: jax.Array
    mu: jax.Array
    sigma: jax.Array
    advantages: jax.Array
    returns: jax.Array


def sample_action(rng, mean, std):
    """Sample an action from a diagonal Gaussian policy."""
    return mean + std * jax.random.normal(rng, mean.shape)


def log_prob_normal(x, mean, std):
    """Elementwise log probability of a normal distribution."""
    var = std**2
    log_scale = jnp.log(std)
    return -((x - mean) ** 2) / (2 * var) - log_scale - 0.5 * jnp.log(2 * jnp.pi)


def compute_gae(rewards, values, dones, gamma, lam):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = jnp.zeros_like(values[0])

    for t in reversed(range(len(rewards))):
        next_value = jnp.zeros_like(values[t]) if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = jnp.stack(advantages)
    returns = advantages + jnp.stack(values)
    return advantages, returns


def ppo_loss(
    params,
    network,
    obs,
    critic_obs,
    actions,
    old_log_probs,
    old_values,
    old_mu,
    old_sigma,
    advantages,
    returns,
    clip_param,
    vf_coef,
    entropy_coef,
):
    """PPO clipped objective with asymmetric critic observations."""
    action_mean, action_std, values = network.apply(params, obs, critic_obs)

    log_probs = jnp.sum(log_prob_normal(actions, action_mean, action_std), axis=-1)
    ratio = jnp.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - clip_param, 1 + clip_param) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    value_pred_clipped = old_values + jnp.clip(values - old_values, -clip_param, clip_param)
    value_loss1 = jnp.square(values - returns)
    value_loss2 = jnp.square(value_pred_clipped - returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))

    entropy = 0.5 * (1 + jnp.log(2 * jnp.pi)) + jnp.log(action_std)
    entropy = jnp.mean(jnp.sum(entropy, axis=-1))
    loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

    kl = jnp.sum(
        jnp.log(action_std / old_sigma)
        + (old_sigma**2 + (old_mu - action_mean) ** 2) / (2 * action_std**2)
        - 0.5,
        axis=-1,
    )

    metrics = {
        "total_loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "mean_kl": jnp.mean(kl),
        "mean_std": jnp.mean(action_std),
    }
    return loss, metrics


def save_checkpoint(path, state, iteration, reward_stats=None):
    """Save trainer checkpoint."""
    checkpoint = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
        "rng": np.array(state.rng),
        "iteration": iteration,
    }
    if reward_stats:
        checkpoint["reward_stats"] = reward_stats

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(path):
    """Load trainer checkpoint."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _flatten_rollout(xs):
    stacked = jnp.stack(xs, axis=0)
    if stacked.ndim == 3:
        return stacked.reshape(-1, stacked.shape[-1])
    return stacked.reshape(-1)


def _debug_terrain_info(env, iteration: int = None):
    """打印地形信息调试函数"""
    print("\n" + "-" * 60)
    print("[Terrain Debug]")
    
    # 检查地形配置
    if hasattr(env, '_terrain_config'):
        print(f"  mesh_type: {env._terrain_config.mesh_type}")
    else:
        print("  ⚠️ env has no _terrain_config")
    
    # 检查地形高度场
    if hasattr(env, '_terrain_height_field'):
        if env._terrain_height_field is not None:
            hf = env._terrain_height_field
            print(f"  height_field shape: {hf.shape}")
            print(f"  height_field range: [{float(hf.min()):.3f}, {float(hf.max()):.3f}]")
            
            # 测试查询函数
            if hasattr(env, '_query_terrain_height'):
                try:
                    h0 = env._query_terrain_height(0.0, 0.0)
                    h1 = env._query_terrain_height(2.0, 2.0)
                    h2 = env._query_terrain_height(-2.0, -2.0)
                    print(f"  query test: h(0,0)={float(h0):.4f}, h(2,2)={float(h1):.4f}, h(-2,-2)={float(h2):.4f}")
                except Exception as e:
                    print(f"  ⚠️ query test failed: {e}")
        else:
            print("  ⚠️ height_field is None")
    else:
        print("  ⚠️ env has no _terrain_height_field")
    
    if iteration is not None:
        print(f"  at iteration: {iteration}")
    print("-" * 60 + "\n")


def train(
    env,
    network,
    num_envs: int = 4096,
    num_steps_per_env: int = 24,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_param: float = 0.2,
    vf_coef: float = 1.0,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
    max_iterations: int = 10000,
    desired_kl: float = 0.01,
    seed: int = 42,
    log_dir: str = None,
    checkpoint_dir: str = None,
    save_interval: int = 50,
    resume_checkpoint: str = None,
):
    """Train an MJX environment with PPO."""
    print("\n" + "=" * 80)
    print("Pure MLP PPO Training")
    print("=" * 80)
    print(f"  num_envs: {num_envs}")
    print(f"  num_steps_per_env: {num_steps_per_env}")
    print(f"  num_learning_epochs: {num_learning_epochs}")
    print(f"  num_mini_batches: {num_mini_batches}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  gamma: {gamma}")
    print(f"  lam: {lam}")
    print(f"  clip_param: {clip_param}")
    print(f"  vf_coef: {vf_coef}")
    print(f"  entropy_coef: {entropy_coef}")
    print(f"  desired_kl: {desired_kl}")
    print("=" * 80 + "\n")

    # ⭐ 打印地形信息（训练开始前）
    _debug_terrain_info(env)

    if checkpoint_dir is None:
        checkpoint_dir = (
            log_dir.replace("tensorboard_", "checkpoints_") if log_dir else "checkpoints/mlp"
        )

    rng = jax.random.PRNGKey(seed)
    obs_size = env.observation_size.get("state", 48)
    critic_obs_size = env.observation_size.get("privileged", 72)
    action_size = env.action_size
    print(f"  Actor obs size: {obs_size}")
    print(f"  Critic obs size: {critic_obs_size}")

    rng, init_rng = jax.random.split(rng)
    params = network.init(
        init_rng,
        jnp.zeros((1, obs_size)),
        jnp.zeros((1, critic_obs_size)),
    )

    current_lr = learning_rate
    min_lr = 1e-5
    max_lr = 1e-3
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=current_lr, eps=1e-5),
    )
    opt_state = optimizer.init(params)
    state = TrainingState(params=params, opt_state=opt_state, step=0, rng=rng)

    start_iteration = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"  Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = load_checkpoint(resume_checkpoint)
        start_iteration = checkpoint["iteration"] + 1
        state = TrainingState(
            params=checkpoint["params"],
            opt_state=checkpoint["opt_state"],
            step=checkpoint["step"],
            rng=jnp.array(checkpoint["rng"]),
        )
        print(f"  Resumed from iteration {start_iteration}")

    writer = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    print("  [1/3] Initializing environments...")
    rng, env_rng = jax.random.split(rng)
    env_rngs = jax.random.split(env_rng, num_envs)

    print("  [2/3] JIT compiling env...")
    jit_reset = jax.jit(jax.vmap(env.reset))
    env_state = jit_reset(env_rngs)
    jax.block_until_ready(env_state.obs["state"])

    step_unclipped = lambda state, action, odom: env.step(state, action, odom, clip_action=False)
    jit_step = jax.jit(jax.vmap(step_unclipped, in_axes=(0, 0, 0)))
    dummy_action = jnp.zeros((num_envs, action_size))
    dummy_odom = jnp.zeros((num_envs, 3))
    compiled_state = jit_step(env_state, dummy_action, dummy_odom)
    jax.block_until_ready(compiled_state)
    print("        ✓ env compiled!")

    print("  [3/3] JIT compiling network...")
    jit_network = jax.jit(network.apply)
    compiled_network = jit_network(
        state.params,
        jnp.zeros((num_envs, obs_size)),
        jnp.zeros((num_envs, critic_obs_size)),
    )
    jax.block_until_ready(compiled_network)
    print("        ✓ network compiled!")

    @jax.jit
    def train_step(params, opt_state, buffer):
        def loss_fn(p):
            return ppo_loss(
                p,
                network,
                buffer.obs,
                buffer.critic_obs,
                buffer.actions,
                buffer.log_probs,
                buffer.values,
                buffer.mu,
                buffer.sigma,
                buffer.advantages,
                buffer.returns,
                clip_param,
                vf_coef,
                entropy_coef,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    print("  JIT compiling train_step...")
    dummy_buffer = RolloutBuffer(
        obs=jnp.zeros((1000, obs_size)),
        critic_obs=jnp.zeros((1000, critic_obs_size)),
        actions=jnp.zeros((1000, action_size)),
        values=jnp.zeros(1000),
        log_probs=jnp.zeros(1000),
        mu=jnp.zeros((1000, action_size)),
        sigma=jnp.ones((1000, action_size)),
        advantages=jnp.zeros(1000),
        returns=jnp.zeros(1000),
    )
    compiled_train = train_step(state.params, state.opt_state, dummy_buffer)
    jax.block_until_ready(compiled_train)
    print("        ✓ train_step compiled!")

    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = jnp.zeros(num_envs)
    cur_episode_length = jnp.zeros(num_envs)

    print("\n" + "=" * 80)
    print("Starting training loop...")
    print(f"  Training from iteration {start_iteration} to {start_iteration + max_iterations}")
    print("=" * 80)
    start_time = time.time()

    for iteration in range(start_iteration, start_iteration + max_iterations):
        iter_start = time.time()
        obs_buffer = []
        critic_obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        dones_buffer = []
        values_buffer = []
        log_probs_buffer = []
        mu_buffer = []
        sigma_buffer = []
        ep_reward_dump = []
        ep_length_dump = []

        for _ in range(num_steps_per_env):
            actor_obs = env_state.obs["state"]
            critic_obs = env_state.obs.get("privileged", actor_obs)

            action_mean, action_std, value = jit_network(state.params, actor_obs, critic_obs)

            rng, action_rng = jax.random.split(state.rng)
            state = state.replace(rng=rng)
            action = sample_action(action_rng, action_mean, action_std)
            action = jnp.where(jnp.isfinite(action), action, 0.0)
            log_prob = jnp.sum(log_prob_normal(action, action_mean, action_std), axis=-1)

            obs_buffer.append(actor_obs)
            critic_obs_buffer.append(critic_obs)
            actions_buffer.append(action)
            values_buffer.append(value)
            log_probs_buffer.append(log_prob)
            mu_buffer.append(action_mean)
            sigma_buffer.append(jnp.broadcast_to(action_std, action_mean.shape))

            env_state = jit_step(env_state, action, jnp.zeros((num_envs, 3)))
            rewards_buffer.append(env_state.reward)
            dones_buffer.append(env_state.done)

            cur_reward_sum = cur_reward_sum + env_state.reward
            cur_episode_length = cur_episode_length + 1

            done_mask = env_state.done > 0.5
            ep_reward_dump.append(
                jnp.where(done_mask, cur_reward_sum, jnp.nan)
            )
            ep_length_dump.append(
                jnp.where(done_mask, cur_episode_length, jnp.nan)
            )
            cur_reward_sum = jnp.where(done_mask, 0.0, cur_reward_sum)
            cur_episode_length = jnp.where(done_mask, 0.0, cur_episode_length)

        all_ep_rewards = np.asarray(jnp.stack(ep_reward_dump))
        all_ep_lengths = np.asarray(jnp.stack(ep_length_dump))
        finished_mask = ~np.isnan(all_ep_rewards)
        if finished_mask.any():
            rewbuffer.extend(all_ep_rewards[finished_mask].tolist())
            lenbuffer.extend(all_ep_lengths[finished_mask].tolist())

        advantages, returns = compute_gae(rewards_buffer, values_buffer, dones_buffer, gamma, lam)
        buffer = RolloutBuffer(
            obs=_flatten_rollout(obs_buffer),
            critic_obs=_flatten_rollout(critic_obs_buffer),
            actions=_flatten_rollout(actions_buffer),
            values=_flatten_rollout(values_buffer),
            log_probs=_flatten_rollout(log_probs_buffer),
            mu=_flatten_rollout(mu_buffer),
            sigma=_flatten_rollout(sigma_buffer),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
        )
        buffer = buffer.replace(
            advantages=(buffer.advantages - jnp.mean(buffer.advantages))
            / (jnp.std(buffer.advantages) + 1e-8)
        )

        batch_size = buffer.obs.shape[0]
        mini_batch_size = batch_size // num_mini_batches
        total_metrics = {}

        for _ in range(num_learning_epochs):
            rng, perm_rng = jax.random.split(state.rng)
            state = state.replace(rng=rng)
            perm = jax.random.permutation(perm_rng, batch_size)

            for mb in range(num_mini_batches):
                mb_indices = perm[mb * mini_batch_size : (mb + 1) * mini_batch_size]
                mb_buffer = RolloutBuffer(
                    obs=buffer.obs[mb_indices],
                    critic_obs=buffer.critic_obs[mb_indices],
                    actions=buffer.actions[mb_indices],
                    values=buffer.values[mb_indices],
                    log_probs=buffer.log_probs[mb_indices],
                    mu=buffer.mu[mb_indices],
                    sigma=buffer.sigma[mb_indices],
                    advantages=buffer.advantages[mb_indices],
                    returns=buffer.returns[mb_indices],
                )

                new_params, new_opt_state, metrics = train_step(
                    state.params, state.opt_state, mb_buffer
                )
                state = state.replace(
                    params=new_params,
                    opt_state=new_opt_state,
                    step=state.step + 1,
                )

                for key, value in metrics.items():
                    total_metrics.setdefault(key, []).append(float(value))

        for key in total_metrics:
            total_metrics[key] = statistics.mean(total_metrics[key])

        mean_kl = total_metrics.get("mean_kl", 0.0)
        old_lr = current_lr
        if mean_kl > desired_kl * 2.0:
            current_lr = max(current_lr / 1.5, min_lr)
        elif mean_kl < desired_kl * 0.5:
            current_lr = min(current_lr * 1.5, max_lr)

        if current_lr != old_lr:
            inner_opt_state = state.opt_state[1]
            if hasattr(inner_opt_state, "hyperparams"):
                inner_opt_state.hyperparams["learning_rate"] = jnp.array(current_lr)

        iter_time = time.time() - iter_start
        
        # ⭐ 每 100 轮打印地形信息
        if iteration % 100 == 0 and iteration > 0:
            _debug_terrain_info(env, iteration)
        
        if iteration % 10 == 0:
            mean_reward = statistics.mean(rewbuffer) if rewbuffer else 0.0
            mean_length = statistics.mean(lenbuffer) if lenbuffer else 0.0
            mean_std = total_metrics.get("mean_std", 0.0)
            print(
                f"Iter {iteration:5d} | reward: {mean_reward:7.2f} | "
                f"len: {mean_length:6.1f} | std: {mean_std:.3f} | "
                f"kl: {mean_kl:.4f} | lr: {current_lr:.2e} | time: {iter_time:.2f}s"
            )

            if writer:
                writer.add_scalar("train/mean_reward", mean_reward, iteration)
                writer.add_scalar("train/mean_episode_length", mean_length, iteration)
                writer.add_scalar("train/mean_std", mean_std, iteration)
                writer.add_scalar("train/mean_kl", mean_kl, iteration)
                writer.add_scalar("train/learning_rate", current_lr, iteration)
                for key, value in total_metrics.items():
                    writer.add_scalar(f"loss/{key}", value, iteration)

        if iteration % save_interval == 0 and iteration > start_iteration:
            checkpoint_path = os.path.join(checkpoint_dir, f"iteration_{iteration}.pkl")
            save_checkpoint(checkpoint_path, state, iteration)

    checkpoint_path = os.path.join(
        checkpoint_dir, f"iteration_{start_iteration + max_iterations}.pkl"
    )
    save_checkpoint(checkpoint_path, state, start_iteration + max_iterations)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 3600:.2f} hours")

    if writer:
        writer.close()

    return state


__all__ = [
    "RolloutBuffer",
    "TrainingState",
    "compute_gae",
    "load_checkpoint",
    "log_prob_normal",
    "ppo_loss",
    "sample_action",
    "save_checkpoint",
    "train",
]