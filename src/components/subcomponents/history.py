"""State history bookkeeping sub-component for MJX environments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp


class HistorySubComponent:
    """Observation history buffer sub-component."""

    def __init__(self, *, default_pose: jax.Array,
                 obs_dof_his_num: int, obs_imu_his_num: int,
                 use_custom_pd: bool):
        self.default_pose = default_pose
        self.obs_dof_his_num = obs_dof_his_num
        self.obs_imu_his_num = obs_imu_his_num
        self.use_custom_pd = use_custom_pd

    def push(self, *, info: dict[str, Any], joint_pos: jax.Array,
             joint_vel: jax.Array, rpy: jax.Array,
             base_ang_vel: jax.Array, torques: jax.Array) -> dict[str, Any]:
        """Push current proprioceptive values into history buffers."""
        return push_observation_history(
            info=info, joint_pos=joint_pos, joint_vel=joint_vel,
            rpy=rpy, base_ang_vel=base_ang_vel, torques=torques,
            use_custom_pd=self.use_custom_pd,
        )

    def update_action(self, *, info: dict[str, Any],
                      action: jax.Array, joint_vel: jax.Array) -> dict[str, Any]:
        """Update last action and joint velocity after reward computation."""
        return update_last_action_state(info=info, action=action, joint_vel=joint_vel)

    def reset_on_done(self, *, info: dict[str, Any],
                      done: jax.Array) -> dict[str, Any]:
        """Reset observation history buffers when an episode terminates."""
        return reset_history_on_done(
            info=info, done=done, default_pose=self.default_pose,
            obs_dof_his_num=self.obs_dof_his_num,
            obs_imu_his_num=self.obs_imu_his_num,
        )


def push_observation_history(
    *,
    info: dict[str, Any],
    joint_pos: jax.Array,
    joint_vel: jax.Array,
    rpy: jax.Array,
    base_ang_vel: jax.Array,
    torques: jax.Array,
    use_custom_pd: bool,
) -> dict[str, Any]:
    """Push current proprioceptive values into observation history buffers."""
    dof_pos_history = jp.roll(info["dof_pos_history"], shift=1, axis=1)
    dof_pos_history = dof_pos_history.at[:, 0].set(joint_pos)

    dof_vel_history = jp.roll(info["dof_vel_history"], shift=1, axis=1)
    dof_vel_history = dof_vel_history.at[:, 0].set(joint_vel)

    rpy_history = jp.roll(info["rpy_history"], shift=1, axis=1)
    rpy_history = rpy_history.at[:, 0].set(rpy)

    ang_lin_history = jp.roll(info["ang_lin_history"], shift=1, axis=1)
    ang_lin_history = ang_lin_history.at[:, 0].set(base_ang_vel)

    info["dof_pos_history"] = dof_pos_history
    info["dof_vel_history"] = dof_vel_history
    info["rpy_history"] = rpy_history
    info["ang_lin_history"] = ang_lin_history

    if not use_custom_pd:
        torque_his = jp.roll(info["torque_his"], shift=1, axis=1)
        info["torque_his"] = torque_his.at[:, 0].set(torques)

    return info


def update_last_action_state(
    *,
    info: dict[str, Any],
    action: jax.Array,
    joint_vel: jax.Array,
) -> dict[str, Any]:
    """Update last action and joint velocity after reward computation."""
    info["last_last_action"] = info["last_action"]
    info["last_action"] = action
    info["last_last_dof_vel"] = info["last_dof_vel"]
    info["last_dof_vel"] = joint_vel
    return info


def reset_history_on_done(
    *,
    info: dict[str, Any],
    done: jax.Array,
    default_pose: jax.Array,
    obs_dof_his_num: int,
    obs_imu_his_num: int,
) -> dict[str, Any]:
    """Reset observation history buffers when an episode terminates."""
    done_expanded = done[..., None] if done.ndim > 0 else done
    info["dof_pos_history"] = jp.where(
        done_expanded,
        jp.ones((12, obs_dof_his_num + 1)) * default_pose[:, None],
        info["dof_pos_history"],
    )
    info["dof_vel_history"] = jp.where(
        done_expanded,
        jp.zeros((12, obs_dof_his_num + 1)),
        info["dof_vel_history"],
    )
    info["rpy_history"] = jp.where(
        done_expanded,
        jp.zeros((3, obs_imu_his_num + 1)),
        info["rpy_history"],
    )
    info["ang_lin_history"] = jp.where(
        done_expanded,
        jp.zeros((3, obs_imu_his_num + 1)),
        info["ang_lin_history"],
    )
    return info


__all__ = [
    "HistorySubComponent",
    "push_observation_history",
    "reset_history_on_done",
    "update_last_action_state",
]
