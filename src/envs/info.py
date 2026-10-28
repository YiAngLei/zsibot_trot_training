"""Typed info schema for the environment state bus.

Uses ``flax.struct.dataclass`` for full JAX pytree compatibility (jit, vmap,
scan).  All fields are ``jax.Array`` (or nested struct-dataclasses) except
``rewards`` which is a dynamic-keyed dict used only for metrics outside JIT.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jp
from flax import struct


# ------------------------------------------------------------------
# Nested sub-structures (logical grouping of info fields)
# ------------------------------------------------------------------


@struct.dataclass
class ContactState:
    """Foot contact bookkeeping."""
    last_contact: jp.ndarray       # (4,) bool
    feet_air_time: jp.ndarray      # (4,)
    feet_contact_time: jp.ndarray  # (4,)
    peak_heights: jp.ndarray       # (4,)


@struct.dataclass
class HistoryBuffers:
    """Observation history ring-buffers (column 0 = most recent)."""
    dof_pos_history: jp.ndarray    # (12, N+1)
    dof_vel_history: jp.ndarray    # (12, N+1)
    rpy_history: jp.ndarray        # (3, N+1)
    ang_lin_history: jp.ndarray    # (3, N+1)
    torque_his: jp.ndarray        # (12, N+1)


@struct.dataclass
class PerturbationState:
    """Push-FM perturbation state machine."""
    push_FM_active: jp.ndarray     # () bool
    push_FM_step_count: jp.ndarray # () int32
    push_force: jp.ndarray         # (3,)
    push_torque: jp.ndarray        # (3,)


# ------------------------------------------------------------------
# Top-level EnvInfo
# ------------------------------------------------------------------


@struct.dataclass
class EnvInfo:
    """Fully typed environment info bus.

    JAX-compatible via ``flax.struct.dataclass`` (registered pytree).
    Use ``EnvInfo.zero(...)`` for zero-initialization.
    """

    # --- Counters ---
    step: jp.ndarray                          # () int32
    common_step_counter: jp.ndarray           # () int32

    # --- Command ---
    command: jp.ndarray                       # (3,)
    heading_target: jp.ndarray                # ()

    # --- Domain Randomization ---
    motor_strength: jp.ndarray                # (12,)
    kp_factor: jp.ndarray                     # (12,)
    kd_factor: jp.ndarray                     # (12,)
    dof_damping: jp.ndarray                   # (12,)
    friction_coeff: jp.ndarray                # ()
    inertia_factors: jp.ndarray               # (1, 3)
    mass_factors: jp.ndarray                  # (1,)
    com_offset: jp.ndarray                    # (3,)
    restitution_coeff: jp.ndarray             # ()
    feet_friction: jp.ndarray                 # ()
    action_latency: jp.ndarray                # () int32

    # --- Contact ---
    contact: ContactState

    # --- History ---
    history: HistoryBuffers

    # --- Perturbation ---
    perturbation: PerturbationState

    # --- Control (previous-step state) ---
    last_action: jp.ndarray                   # (12,)
    last_last_action: jp.ndarray              # (12,)
    last_dof_vel: jp.ndarray                  # (12,)
    last_last_dof_vel: jp.ndarray             # (12,)
    last_base_lin_vel: jp.ndarray             # (3,)

    # --- RNG ---
    rng: jp.ndarray                           # (2,) uint32

    # --- Termination flags ---
    done_fell_over: jp.ndarray                # () bool
    done_illegal_contact: jp.ndarray          # () bool
    done_nan: jp.ndarray                      # () bool
    time_out: jp.ndarray                      # () bool
    done_unflip: jp.ndarray                   # () bool
    unflip_time_out: jp.ndarray               # () int32
    speed_unfollow_time_out: jp.ndarray       # () int32
    low_height_time_out: jp.ndarray           # () int32
    collision_time_out: jp.ndarray            # () int32
    ever_flipped: jp.ndarray                  # () bool

    # --- Odometry ---
    odom_data: jp.ndarray                     # (3,)
    odom_ground_truth: jp.ndarray             # (3,)

    # --- Reward (dynamic-keyed, kept as plain dict for metrics) ---
    rewards: dict                             # dict[str, jp.ndarray]

    # --- Cumulative accumulators ---
    cumulative_speed: jp.ndarray              # ()
    cumulative_cmd_speed: jp.ndarray          # ()


__all__ = [
    "ContactState",
    "EnvInfo",
    "HistoryBuffers",
    "PerturbationState",
]
