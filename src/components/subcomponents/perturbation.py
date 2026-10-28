"""Velocity kicks and force/torque perturbations for sim-to-real robustness."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp


class PerturbationSubComponent:
    """Periodic velocity kicks and continuous force/torque pushes.

    Two independent perturbations, matching Isaac Gym:

    - **velocity kick** (`push_robots`): every `push_interval` steps, add a
      uniform random horizontal velocity to the base.
    - **force/torque** (`push_robots_FM`): state machine that holds a
      random force + torque for `push_fm_duration` consecutive steps
      every `push_fm_interval` steps.

    All perturbation state lives in `info`; the sub-component is stateless
    aside from configuration.
    """

    def __init__(
        self,
        *,
        push_robots: bool = False,
        push_interval: int = 500,
        max_push_vel_xy: float = 0.5,
        push_robots_fm: bool = False,
        push_fm_interval: int = 500,
        push_fm_duration: int = 200,
        max_push_force_fm: float = 50.0,
        max_push_torque_fm: float = 10.0,
    ):
        self.push_robots = push_robots
        self.push_interval = push_interval
        self.max_push_vel_xy = max_push_vel_xy
        self.push_robots_fm = push_robots_fm
        self.push_fm_interval = push_fm_interval
        self.push_fm_duration = push_fm_duration
        self.max_push_force_fm = max_push_force_fm
        self.max_push_torque_fm = max_push_torque_fm

    def apply(
        self,
        *,
        data: Any,
        info: dict[str, Any],
        qfrc_applied: jax.Array | None,
        model_nv: int,
        kick_rng: jax.Array,
        push_rng: jax.Array,
    ) -> tuple[Any, jax.Array, dict[str, Any]]:
        """Apply velocity kick and/or FM force/torque. Returns (data, qfrc_applied, info)."""
        if self.push_robots:
            data = self._apply_velocity_kick(data, info, kick_rng)

        if self.push_robots_fm:
            qfrc_applied, info = self._apply_fm_force(qfrc_applied, info, model_nv, push_rng)

        return data, qfrc_applied, info

    # ------------------------------------------------------------------
    # Internal: periodic velocity kick
    # ------------------------------------------------------------------

    def _apply_velocity_kick(
        self, data: Any, info: dict[str, Any], rng: jax.Array
    ) -> Any:
        should_push = (info["common_step_counter"] % self.push_interval) == 0
        kick_vel = jax.random.uniform(
            rng, (2,),
            minval=-self.max_push_vel_xy,
            maxval=self.max_push_vel_xy,
        )
        kick_vel = jp.where(should_push, kick_vel, jp.zeros(2))
        qvel = data.qvel
        qvel = qvel.at[:2].set(qvel[:2] + kick_vel)
        return data.replace(qvel=qvel)

    # ------------------------------------------------------------------
    # Internal: stateful force/torque machine
    # ------------------------------------------------------------------

    def _apply_fm_force(
        self,
        qfrc_applied: jax.Array | None,
        info: dict[str, Any],
        model_nv: int,
        rng: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        if qfrc_applied is None:
            qfrc_applied = jp.zeros(model_nv)

        push_FM_active = info["push_FM_active"]
        push_FM_step_count = info["push_FM_step_count"]
        push_force = info["push_force"]
        push_torque = info["push_torque"]

        should_start = (info["common_step_counter"] % self.push_fm_interval) == 0
        should_start = should_start & (info["common_step_counter"] > 0)

        rng, force_rng, torque_rng = jax.random.split(rng, 3)
        new_force = jax.random.uniform(
            force_rng, (3,),
            minval=-self.max_push_force_fm,
            maxval=self.max_push_force_fm,
        )
        new_torque = jax.random.uniform(
            torque_rng, (3,),
            minval=-self.max_push_torque_fm,
            maxval=self.max_push_torque_fm,
        )

        push_force = jp.where(should_start, new_force, push_force)
        push_torque = jp.where(should_start, new_torque, push_torque)
        push_FM_active = jp.where(should_start, True, push_FM_active)
        push_FM_step_count = jp.where(should_start, 0, push_FM_step_count)

        should_stop = push_FM_step_count >= self.push_fm_duration
        push_FM_active = jp.where(should_stop, False, push_FM_active)

        active_force = jp.where(push_FM_active, push_force, jp.zeros(3))
        active_torque = jp.where(push_FM_active, push_torque, jp.zeros(3))
        qfrc_applied = qfrc_applied.at[:3].set(active_force)
        qfrc_applied = qfrc_applied.at[3:6].set(active_torque)

        push_FM_step_count = jp.where(
            push_FM_active, push_FM_step_count + 1, push_FM_step_count
        )

        info["push_FM_active"] = push_FM_active
        info["push_FM_step_count"] = push_FM_step_count
        info["push_force"] = push_force
        info["push_torque"] = push_torque
        return qfrc_applied, info


__all__ = ["PerturbationSubComponent"]
