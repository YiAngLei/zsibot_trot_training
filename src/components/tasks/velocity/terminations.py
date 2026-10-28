"""Velocity task termination conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jp


@dataclass(frozen=True)
class TerminationTermSpec:
    name: str
    description: str


VELOCITY_TERMINATION_TERMS: Mapping[str, TerminationTermSpec] = {
    "fell_over": TerminationTermSpec("fell_over", "Projected gravity exceeds task tilt limit."),
    "illegal_contact": TerminationTermSpec("illegal_contact", "Non-foot body contacts the ground."),
    "nan": TerminationTermSpec("nan", "Physics state contains non-finite values."),
    "timeout": TerminationTermSpec("timeout", "Episode reached configured length."),
}


def compute_velocity_terminations(
    *,
    data,
    projected_gravity,
    nonfoot_body_ids,
    illegal_contact_force_threshold,
    fell_over_limit_angle,
    step,
    episode_length,
):
    """Compute velocity task termination flags."""
    limit_cos = jp.cos(fell_over_limit_angle)
    done_fell_over = projected_gravity[2] > -limit_cos

    nonfoot_forces = data.cfrc_ext[nonfoot_body_ids, 3:6]
    nonfoot_force_mag = jp.linalg.norm(nonfoot_forces, axis=-1)
    done_illegal_contact = jp.any(nonfoot_force_mag > illegal_contact_force_threshold)

    is_finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
    done_nan = ~is_finite
    time_out = step >= episode_length

    done = done_fell_over | done_illegal_contact | done_nan | time_out
    return {
        "done": done,
        "done_fell_over": done_fell_over,
        "done_illegal_contact": done_illegal_contact,
        "done_nan": done_nan,
        "time_out": time_out,
    }


def write_termination_info(info, terminations):
    """Store termination flags and compatibility aliases in info."""
    done = terminations["done"]
    done_fell_over = terminations["done_fell_over"]

    info["done_fell_over"] = done_fell_over
    info["done_illegal_contact"] = terminations["done_illegal_contact"]
    info["done_nan"] = terminations["done_nan"]
    info["time_out"] = terminations["time_out"]

    info["done_unflip"] = done_fell_over
    info["unflip_time_out"] = jp.where(done, 0, info.get("unflip_time_out", 0))
    info["speed_unfollow_time_out"] = jp.where(
        done, 0, info.get("speed_unfollow_time_out", 0)
    )
    info["low_height_time_out"] = jp.where(done, 0, info.get("low_height_time_out", 0))
    info["collision_time_out"] = jp.where(done, 0, info.get("collision_time_out", 0))
    return info
