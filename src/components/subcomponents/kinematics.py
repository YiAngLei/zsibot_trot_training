"""Kinematic state extraction helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from brax import math


def extract_base_state(
    data: Any,
    mjx_to_policy_idx: jax.Array,
    num_joints: int = 12,
) -> dict[str, jax.Array]:
    """Project world-frame velocities into body frame and reindex joints.

    Returns a dict with the 8 quantities step() needs after physics:
    base_pos, base_quat, base_lin_vel, base_ang_vel, projected_gravity,
    joint_pos, joint_vel, torques.

    Conventions:
    - qpos[0:3] base position, qpos[3:7] base quaternion (w,x,y,z)
    - qpos[7:7+num_joints] joint positions in MJX order
    - qvel[0:3] base linear velocity in world frame
    - qvel[3:6] base angular velocity in world frame
    - qvel[6:6+num_joints] joint velocities in MJX order
    """
    base_quat = data.qpos[3:7]
    inv_rot = math.quat_inv(base_quat)

    base_pos = data.qpos[:3]
    base_lin_vel = math.rotate(data.qvel[:3], inv_rot)
    base_ang_vel = math.rotate(data.qvel[3:6], inv_rot)
    projected_gravity = math.rotate(jp.array([0.0, 0.0, -1.0]), inv_rot)

    mjx_joint_pos = data.qpos[7:7 + num_joints]
    mjx_joint_vel = data.qvel[6:6 + num_joints]
    joint_pos = mjx_joint_pos[mjx_to_policy_idx]
    joint_vel = mjx_joint_vel[mjx_to_policy_idx]

    return {
        "base_pos": base_pos,
        "base_quat": base_quat,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def quat_to_rpy(quat: jax.Array) -> jax.Array:
    """Convert a (w,x,y,z) quaternion to (roll, pitch, yaw) in radians.

    Uses the standard Tait-Bryan decomposition with guarded arcsin to avoid
    the singularity at |pitch| = pi/2.
    """
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = jp.where(jp.abs(sinp) >= 1, jp.sign(sinp) * jp.pi / 2, jp.arcsin(sinp))
    yaw = jp.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return jp.array([roll, pitch, yaw])


def get_torques(
    data: Any,
    computed_torques_mjx: jax.Array | None,
    torque_limits: jax.Array,
    mjx_to_policy_idx: jax.Array,
) -> jax.Array:
    """Recover the applied joint torques in policy/Isaac order, clipped.

    Prefers the explicitly-computed torques (custom PD branch); falls back
    to data.actuator_force for built-in PD.
    """
    if computed_torques_mjx is not None:
        torques_mjx = computed_torques_mjx
    else:
        torques_mjx = data.actuator_force
    torques = torques_mjx[mjx_to_policy_idx]
    return jp.clip(torques, -torque_limits, torque_limits)


__all__ = ["extract_base_state", "get_torques", "quat_to_rpy"]
