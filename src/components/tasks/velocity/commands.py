"""Velocity command sampling."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp


def sample_velocity_command(
    rng: jax.Array,
    command_config: Any,
    heading_command: bool,
) -> tuple[jax.Array, jax.Array]:
    """Sample the current velocity command.

    Returns:
        ``(command, heading_target)`` where command is ``[vx, vy, wz]``.
    """
    # Match the original env's RNG consumption exactly. It splits 5 keys
    # here even though only the first 4 are used downstream.
    rng1, rng2, rng3, rng4, _ = jax.random.split(rng, 5)
    ranges = command_config.ranges

    vx = jax.random.uniform(rng1, minval=ranges.lin_vel_x[0], maxval=ranges.lin_vel_x[1])
    vy = jax.random.uniform(rng2, minval=ranges.lin_vel_y[0], maxval=ranges.lin_vel_y[1])

    if heading_command:
        heading_range = getattr(ranges, "heading", (-3.14, 3.14))
        heading_target = jax.random.uniform(
            rng3,
            minval=heading_range[0],
            maxval=heading_range[1],
        )
        wz = jp.array(0.0)
    else:
        wz = jax.random.uniform(
            rng3,
            minval=ranges.ang_vel_yaw[0],
            maxval=ranges.ang_vel_yaw[1],
        )
        heading_target = jp.array(0.0)

    zero_cmd_prob = getattr(command_config, "zero_cmd_probability", 0.15)
    zero_prob = jax.random.uniform(rng4)
    command = jp.where(zero_prob < zero_cmd_prob, jp.zeros(3), jp.array([vx, vy, wz]))
    return command, heading_target
