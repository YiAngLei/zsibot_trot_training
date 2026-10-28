"""Velocity task reward registry and extracted reward terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jp


RewardFn = Callable[["VelocityRewardContext", Any], jax.Array]


@dataclass(frozen=True)
class VelocityRewardContext:
    """Inputs shared by the first registry-backed velocity reward terms."""

    command: jax.Array
    action: jax.Array
    last_action: jax.Array
    joint_pos: jax.Array
    default_pose: jax.Array
    soft_joint_lower: jax.Array
    soft_joint_upper: jax.Array
    base_lin_vel: jax.Array
    base_ang_vel: jax.Array
    projected_gravity: jax.Array
    data: Any | None = None
    feet_body_ids: jax.Array | None = None
    feet_air_time: jax.Array | None = None
    first_contact: jax.Array | None = None


@dataclass(frozen=True)
class RewardTermSpec:
    name: str
    description: str
    fn: RewardFn | None = None

    @property
    def extracted(self) -> bool:
        return self.fn is not None


def tracking_lin_vel(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Track commanded planar velocity and damp vertical base velocity."""
    tracking_sigma_lin = getattr(reward_config, "tracking_sigma_lin", 0.5)
    xy_error = jp.sum(jp.square(context.command[:2] - context.base_lin_vel[:2]))
    z_error = jp.square(context.base_lin_vel[2])
    return jp.exp(-(xy_error + z_error) / (tracking_sigma_lin**2))


def tracking_ang_vel(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Track commanded yaw velocity and damp roll/pitch angular velocity."""
    tracking_sigma_ang = getattr(reward_config, "tracking_sigma_ang", 0.707)
    z_error = jp.square(context.command[2] - context.base_ang_vel[2])
    xy_error = jp.sum(jp.square(context.base_ang_vel[:2]))
    return jp.exp(-(z_error + xy_error) / (tracking_sigma_ang**2))


def lateral_vel(context: VelocityRewardContext, _: Any) -> jax.Array:
    """Penalize body-frame lateral velocity drift."""
    return jp.square(context.base_lin_vel[1])


def yaw_rate_error(context: VelocityRewardContext, _: Any) -> jax.Array:
    """Penalize direct yaw-rate tracking error."""
    return jp.square(context.command[2] - context.base_ang_vel[2])


def upright(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Keep projected gravity close to the upright axis."""
    upright_sigma = getattr(reward_config, "upright_sigma", 0.447)
    xy_squared = jp.sum(jp.square(context.projected_gravity[:2]))
    return jp.exp(-xy_squared / (upright_sigma**2))


def dof_pos_limits(context: VelocityRewardContext, _: Any) -> jax.Array:
    """Penalize excursions beyond the configured soft joint bounds."""
    out_of_limits = -jp.clip(context.joint_pos - context.soft_joint_lower, max=0.0)
    out_of_limits += jp.clip(context.joint_pos - context.soft_joint_upper, min=0.0)
    return jp.sum(out_of_limits)


def action_rate(context: VelocityRewardContext, _: Any) -> jax.Array:
    """Penalize policy action changes from the previous step."""
    return jp.sum(jp.square(context.action - context.last_action))


def pose(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Reward staying near the velocity-dependent nominal posture."""
    linear_speed = jp.linalg.norm(context.command[:2])
    angular_speed = jp.abs(context.command[2])
    total_speed = linear_speed + angular_speed

    walking_threshold = getattr(reward_config, "pose_walking_threshold", 0.05)
    running_threshold = getattr(reward_config, "pose_running_threshold", 1.5)

    is_standing = total_speed < walking_threshold
    is_running = total_speed >= running_threshold

    std_standing_hip = getattr(reward_config, "pose_std_standing_hip", 0.05)
    std_standing_calf = getattr(reward_config, "pose_std_standing_calf", 0.1)
    std_walking_hip = getattr(reward_config, "pose_std_walking_hip", 0.3)
    std_walking_calf = getattr(reward_config, "pose_std_walking_calf", 0.6)
    std_running_hip = getattr(reward_config, "pose_std_running_hip", 0.3)
    std_running_calf = getattr(reward_config, "pose_std_running_calf", 0.6)

    std_hip = jp.where(
        is_standing, std_standing_hip, jp.where(is_running, std_running_hip, std_walking_hip)
    )
    std_calf = jp.where(
        is_standing,
        std_standing_calf,
        jp.where(is_running, std_running_calf, std_walking_calf),
    )

    joint_error = context.joint_pos - context.default_pose
    hip_indices = jp.array([0, 1, 3, 4, 6, 7, 9, 10])
    calf_indices = jp.array([2, 5, 8, 11])

    hip_term = jp.sum(jp.square(joint_error[hip_indices]) / (std_hip**2))
    calf_term = jp.sum(jp.square(joint_error[calf_indices]) / (std_calf**2))
    return jp.exp(-((hip_term + calf_term) / 12.0))


def feet_air_time(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Reward feet whose current air time is in the configured target range."""
    threshold_min = getattr(reward_config, "feet_air_time_target_min", 0.05)
    threshold_max = getattr(reward_config, "feet_air_time_target_max", 0.5)
    command_threshold = getattr(reward_config, "command_threshold", 0.5)

    current_air_time = context.feet_air_time
    if current_air_time is None:
        current_air_time = jp.zeros(4)

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    rew_air = jp.sum(in_range.astype(jp.float32))
    total_command = jp.linalg.norm(context.command[:2]) + jp.abs(context.command[2])
    return rew_air * (total_command > command_threshold).astype(jp.float32)


def foot_clearance(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Penalize swing-foot height error weighted by planar foot velocity."""
    if context.data is None or context.feet_body_ids is None:
        return jp.array(0.0)

    target_height = getattr(reward_config, "foot_clearance_target", 0.1)
    command_threshold = getattr(reward_config, "command_threshold", 0.05)

    foot_z = context.data.xpos[context.feet_body_ids, 2]
    feet_vel = context.data.cvel[context.feet_body_ids, 3:6]
    vel_norm = jp.linalg.norm(feet_vel[:, :2], axis=1)
    cost = jp.sum(jp.abs(foot_z - target_height) * vel_norm)

    total_command = jp.linalg.norm(context.command[:2]) + jp.abs(context.command[2])
    return cost * (total_command > command_threshold).astype(jp.float32)


def slip(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Penalize planar foot velocity while the foot is in contact."""
    if context.data is None or context.feet_body_ids is None:
        return jp.array(0.0)

    command_threshold = getattr(reward_config, "command_threshold", 0.05)
    feet_forces = context.data.cfrc_ext[context.feet_body_ids, 3:6]
    in_contact = (feet_forces[:, 2] > 1.0).astype(jp.float32)
    feet_vel = context.data.cvel[context.feet_body_ids, 3:6]
    vel_xy_norm_sq = jp.square(jp.linalg.norm(feet_vel[:, :2], axis=1))
    cost = jp.sum(vel_xy_norm_sq * in_contact)

    total_command = jp.linalg.norm(context.command[:2]) + jp.abs(context.command[2])
    return cost * (total_command > command_threshold).astype(jp.float32)


def soft_landing(context: VelocityRewardContext, reward_config: Any) -> jax.Array:
    """Penalize first-contact impact magnitude."""
    if context.data is None or context.feet_body_ids is None or context.first_contact is None:
        return jp.array(0.0)

    command_threshold = getattr(reward_config, "command_threshold", 0.05)
    feet_forces = context.data.cfrc_ext[context.feet_body_ids, 3:6]
    force_magnitude = jp.linalg.norm(feet_forces, axis=1)
    cost = jp.sum(force_magnitude * context.first_contact.astype(jp.float32))

    total_command = jp.linalg.norm(context.command[:2]) + jp.abs(context.command[2])
    return cost * (total_command > command_threshold).astype(jp.float32)


def compute_foot_swing_height_reward(
    *,
    data: Any,
    feet_body_ids: jax.Array,
    contact_filt: jax.Array,
    first_contact: jax.Array,
    peak_heights: jax.Array,
    command: jax.Array,
    reward_config: Any,
) -> tuple[jax.Array, jax.Array]:
    """Compute swing-height penalty and the next peak-height state."""
    target_height = getattr(reward_config, "foot_clearance_target", 0.1)
    command_threshold = getattr(reward_config, "command_threshold", 0.05)

    foot_heights = data.xpos[feet_body_ids, 2]
    in_air = contact_filt == False
    next_peak_heights = jp.where(in_air, jp.maximum(peak_heights, foot_heights), peak_heights)

    error = next_peak_heights / target_height - 1.0
    cost = jp.sum(jp.square(error) * first_contact.astype(jp.float32))

    total_command = jp.linalg.norm(command[:2]) + jp.abs(command[2])
    active = (total_command > command_threshold).astype(jp.float32)
    reward = cost * active
    next_peak_heights = jp.where(first_contact, jp.zeros(4), next_peak_heights)
    return reward, next_peak_heights


VELOCITY_REWARD_TERMS: Mapping[str, RewardTermSpec] = {
    "tracking_lin_vel": RewardTermSpec(
        "tracking_lin_vel",
        "Track commanded linear velocity.",
        tracking_lin_vel,
    ),
    "tracking_ang_vel": RewardTermSpec(
        "tracking_ang_vel",
        "Track commanded yaw velocity.",
        tracking_ang_vel,
    ),
    "lateral_vel": RewardTermSpec(
        "lateral_vel",
        "Penalize body-frame lateral drift.",
        lateral_vel,
    ),
    "yaw_rate_error": RewardTermSpec(
        "yaw_rate_error",
        "Penalize direct yaw-rate tracking error.",
        yaw_rate_error,
    ),
    "upright": RewardTermSpec(
        "upright",
        "Keep projected gravity close to upright.",
        upright,
    ),
    "pose": RewardTermSpec(
        "pose",
        "Stay near velocity-dependent nominal posture.",
        pose,
    ),
    "dof_pos_limits": RewardTermSpec(
        "dof_pos_limits",
        "Penalize soft joint limit violations.",
        dof_pos_limits,
    ),
    "action_rate": RewardTermSpec("action_rate", "Penalize action changes.", action_rate),
    "feet_air_time": RewardTermSpec(
        "feet_air_time",
        "Reward valid swing timing.",
        feet_air_time,
    ),
    "foot_clearance": RewardTermSpec(
        "foot_clearance",
        "Penalize swing-foot height error.",
        foot_clearance,
    ),
    "foot_swing_height": RewardTermSpec(
        "foot_swing_height",
        "Penalize peak swing height error.",
    ),
    "slip": RewardTermSpec("slip", "Penalize foot slip in contact.", slip),
    "soft_landing": RewardTermSpec(
        "soft_landing",
        "Penalize high first-contact impact.",
        soft_landing,
    ),
}


def compute_extracted_velocity_rewards(
    *,
    reward_scales: Mapping[str, float],
    reward_config: Any,
    context: VelocityRewardContext,
) -> dict[str, jax.Array]:
    """Compute active registry-backed velocity terms in registry order."""
    rewards = {}
    for name, spec in VELOCITY_REWARD_TERMS.items():
        if spec.fn is not None and name in reward_scales:
            rewards[name] = spec.fn(context, reward_config) * reward_scales[name]
    return rewards
