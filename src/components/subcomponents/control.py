"""Control primitives for MJX legged robots."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx


class ControlSubComponent:
    """Torque control and physics stepping sub-component."""

    def __init__(self, *, use_custom_pd: bool, n_substeps: int,
                 randomize_torques_latency: bool = False,
                 torque_his_num: int = 0,
                 action_scale: float = 0.25,
                 default_pose_mjx: jax.Array | None = None,
                 joint_lower: jax.Array | None = None,
                 joint_upper: jax.Array | None = None,
                 isaac_to_mjx_idx: jax.Array | None = None,
                 mjx_to_isaac_idx: jax.Array | None = None,
                 kp_array: jax.Array | None = None,
                 kd_array: jax.Array | None = None,
                 torque_limits: jax.Array | None = None):
        self.use_custom_pd = use_custom_pd
        self.n_substeps = n_substeps
        self.randomize_torques_latency = randomize_torques_latency
        self.torque_his_num = torque_his_num
        self.action_scale = action_scale
        self.default_pose_mjx = default_pose_mjx
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper
        self.isaac_to_mjx_idx = isaac_to_mjx_idx
        self.mjx_to_isaac_idx = mjx_to_isaac_idx
        self.kp_array = kp_array
        self.kd_array = kd_array
        self.torque_limits = torque_limits

    def step(self, *, model: mjx.Model, data: mjx.Data,
             action_mjx: jax.Array, kp_factor: jax.Array | None = None,
             kd_factor: jax.Array | None = None,
             motor_strength_mjx: jax.Array | None = None,
             dof_damping_mjx: jax.Array | None = None,
             qfrc_applied: jax.Array | None = None,
             info: dict[str, Any] | None = None) -> tuple[Any, jax.Array]:
        """Perform physics simulation step.

        Returns (data, torques) where torques is the MJX-order torque array.
        """
        if self.use_custom_pd:
            target_pos_mjx = self.default_pose_mjx + action_mjx * self.action_scale
            target_pos_mjx = jp.clip(target_pos_mjx, self.joint_lower, self.joint_upper)

            effective_kp = self.kp_array * (kp_factor if kp_factor is not None else 1.0)
            effective_kd = self.kd_array * (kd_factor if kd_factor is not None else 1.0)

            if self.randomize_torques_latency:
                new_torques_mjx = compute_pd_torques(
                    data, target_pos_mjx, effective_kp, effective_kd,
                    self.torque_limits, motor_strength_mjx, dof_damping_mjx,
                )
                data = mjx_step_with_torques(
                    model, data, new_torques_mjx, self.n_substeps, qfrc_applied,
                )
                return data, new_torques_mjx
            else:
                data, torques_mjx = mjx_step_custom_pd(
                    model, data, target_pos_mjx, effective_kp, effective_kd,
                    self.torque_limits, self.n_substeps, qfrc_applied,
                    motor_strength_mjx, dof_damping_mjx,
                )
                return data, torques_mjx
        else:
            motor_targets = action_mjx
            data = mjx_step(model, data, motor_targets, self.n_substeps, qfrc_applied)
            return data, jp.zeros(12)

    def compute_torques_only(self, *, data: mjx.Data, action_mjx: jax.Array,
                             kp_factor: jax.Array | None = None,
                             kd_factor: jax.Array | None = None,
                             motor_strength_mjx: jax.Array | None = None,
                             dof_damping_mjx: jax.Array | None = None) -> jax.Array:
        """Compute PD torques without stepping physics."""
        target_pos_mjx = self.default_pose_mjx + action_mjx * self.action_scale
        target_pos_mjx = jp.clip(target_pos_mjx, self.joint_lower, self.joint_upper)
        effective_kp = self.kp_array * (kp_factor if kp_factor is not None else 1.0)
        effective_kd = self.kd_array * (kd_factor if kd_factor is not None else 1.0)
        return compute_pd_torques(
            data, target_pos_mjx, effective_kp, effective_kd,
            self.torque_limits, motor_strength_mjx, dof_damping_mjx,
        )

    # ------------------------------------------------------------------
    # Latency + history management (extracted from GenericEnv)
    # ------------------------------------------------------------------

    def get_effective_action(self, action: jax.Array, info: dict) -> jax.Array:
        """Apply action-latency substitution.

        If `info['action_latency'] > 0`, replay the previous action; otherwise
        use the new action. Reindex from policy/Isaac order to MJX order.
        """
        action_latency = info.get("action_latency", 0)
        effective = jp.where(action_latency > 0, info["last_action"], action)
        return effective[self.isaac_to_mjx_idx]

    def push_torque_history(self, info: dict, new_torques_mjx: jax.Array) -> dict:
        """Roll the (12, his_num+1) torque history and write new torques at col 0.

        Returns the updated info dict. Operates in policy/Isaac order so
        the history buffer is consistent with what `info['torque_his']`
        already stores.
        """
        torque_his = info["torque_his"]
        torque_his_mjx = torque_his[self.isaac_to_mjx_idx, :]
        torque_his_mjx = jp.roll(torque_his_mjx, shift=1, axis=1)
        torque_his_mjx = torque_his_mjx.at[:, 0].set(new_torques_mjx)
        info["torque_his"] = torque_his_mjx[self.mjx_to_isaac_idx, :]
        return info

    def select_applied_torques(
        self,
        rng: jax.Array,
        info: dict,
        new_torques_mjx: jax.Array,
        current_step: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Choose torques for the latency-enabled step.

        Returns (applied_torques_mjx, new_torques_for_reward) where the
        first is what we feed to the physics step and the second is the
        always-fresh torques (for reward computation).

        Strategy:
        - delayed_idx ~ Uniform[max(0, his_num-2), his_num]
        - If step < delayed_idx, use new_torques (history not yet filled)
        - Else use torque_his[:, delayed_idx]
        """
        torque_his = info["torque_his"]
        torque_his_mjx = torque_his[self.isaac_to_mjx_idx, :]

        rng, latency_rng = jax.random.split(rng)
        info["rng"] = rng

        min_idx = max(0, self.torque_his_num - 2)
        max_idx = self.torque_his_num
        delayed_idx = jax.random.randint(latency_rng, (), min_idx, max_idx + 1)
        delayed_torques = torque_his_mjx[:, delayed_idx]

        use_new = current_step < delayed_idx
        applied = jp.where(use_new, new_torques_mjx, delayed_torques)
        return applied, new_torques_mjx


# ---------------------------------------------------------------------------
# Low-level helper functions (kept as module-level for JAX tracing clarity)
# ---------------------------------------------------------------------------

def mjx_step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
    qfrc_applied: jax.Array | None = None,
) -> mjx.Data:
    """Perform physics simulation step using MuJoCo actuator controls."""

    def single_step(data, _):
        data = data.replace(ctrl=action)
        if qfrc_applied is not None:
            data = data.replace(qfrc_applied=qfrc_applied)
        data = mjx.step(model, data)
        return data, None

    return jax.lax.scan(single_step, data, (), n_substeps)[0]


def mjx_step_custom_pd(
    model: mjx.Model,
    data: mjx.Data,
    target_pos: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    torque_limits: jax.Array,
    n_substeps: int = 1,
    qfrc_applied: jax.Array | None = None,
    motor_strength: jax.Array | None = None,
    dof_damping: jax.Array | None = None,
) -> tuple[mjx.Data, jax.Array]:
    """Step physics with custom PD torques matching the Isaac Gym formula."""
    if motor_strength is None:
        motor_strength = jp.ones(12)
    if dof_damping is None:
        dof_damping = jp.zeros(12)

    final_torques = jp.zeros(12)

    def single_step(carry, _):
        data, _ = carry
        dof_pos = data.qpos[7:19]
        dof_vel = data.qvel[6:18]
        torques = kp * (target_pos - dof_pos) - kd * dof_vel
        torques = torques - dof_damping * dof_vel
        torques = torques * motor_strength
        torques = jp.clip(torques, -torque_limits, torque_limits)

        total_qfrc = jp.zeros(model.nv)
        total_qfrc = total_qfrc.at[6:18].set(torques)
        if qfrc_applied is not None:
            total_qfrc = total_qfrc + qfrc_applied

        data = data.replace(ctrl=jp.zeros(12), qfrc_applied=total_qfrc)
        data = mjx.step(model, data)
        return (data, torques), torques

    (data, final_torques), _ = jax.lax.scan(single_step, (data, final_torques), (), n_substeps)
    return data, final_torques


def mjx_step_with_torques(
    model: mjx.Model,
    data: mjx.Data,
    torques: jax.Array,
    n_substeps: int = 1,
    qfrc_applied: jax.Array | None = None,
) -> mjx.Data:
    """Step physics using a fixed torque vector for every substep."""

    def single_step(data, _):
        total_qfrc = jp.zeros(model.nv)
        total_qfrc = total_qfrc.at[6:18].set(torques)
        if qfrc_applied is not None:
            total_qfrc = total_qfrc + qfrc_applied
        data = data.replace(ctrl=jp.zeros(12), qfrc_applied=total_qfrc)
        data = mjx.step(model, data)
        return data, None

    data, _ = jax.lax.scan(single_step, data, (), n_substeps)
    return data


def compute_pd_torques(
    data: mjx.Data,
    target_pos: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    torque_limits: jax.Array,
    motor_strength: jax.Array | None = None,
    dof_damping: jax.Array | None = None,
) -> jax.Array:
    """Compute PD torques without applying them."""
    if motor_strength is None:
        motor_strength = jp.ones(12)
    if dof_damping is None:
        dof_damping = jp.zeros(12)

    dof_pos = data.qpos[7:19]
    dof_vel = data.qvel[6:18]
    torques = kp * (target_pos - dof_pos) - kd * dof_vel
    torques = torques - dof_damping * dof_vel
    torques = torques * motor_strength
    return jp.clip(torques, -torque_limits, torque_limits)


__all__ = [
    "ControlSubComponent",
    "compute_pd_torques",
    "mjx_step",
    "mjx_step_custom_pd",
    "mjx_step_with_torques",
]
