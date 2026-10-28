"""Velocity tracking task specification."""

from __future__ import annotations

from .base import ObservationSpec, TaskSpec


def build_velocity_task() -> TaskSpec:
    """Build the current velocity task spec."""
    return TaskSpec(
        name="velocity",
        command_dim=3,
        observation=ObservationSpec(actor_size=48, critic_size=72, odom_size=29),
        reward_names=(
            "tracking_lin_vel",
            "tracking_ang_vel",
            "yaw_rate_error",
            "upright",
            "pose",
            "dof_pos_limits",
            "action_rate",
            "feet_air_time",
            "foot_clearance",
            "foot_swing_height",
            "slip",
            "soft_landing",
        ),
        termination_names=("fell_over", "illegal_contact", "nan", "timeout"),
    )
