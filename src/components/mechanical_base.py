"""Mechanical base with default sub-component orchestration.

Provides default implementations for all non-task-specific methods so that
task components only need to implement the 7 task-specific abstracts from
BaseComponent (plus optionally build_initial_info and maybe_resample_command).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as jp

from .base import BaseComponent
from .subcomponents.kinematics import extract_base_state, quat_to_rpy


class MechanicalBase(BaseComponent):
    """Default implementations for all mechanical (non-task-specific) methods.

    Subclasses must set these in ``__init__``:

    Sub-components:
        ``self._control``, ``self._feet``, ``self._history``,
        ``self._physics``, ``self._reset``, ``self._perturbation``

    Scalars / arrays:
        ``self.isaac_to_mjx_idx``, ``self.mjx_to_isaac_idx``,
        ``self.model_nv``, ``self.dt``, ``self.use_custom_pd``,
        ``self.only_positive_rewards``, ``self.termination_reward_scale``
    """

    # ------------------------------------------------------------------
    # Group B: Mechanical facade – default implementations
    # ------------------------------------------------------------------

    def step_physics(self, *, model, data, action, info, **kwargs: Any):
        """Perturbation → DR forces → action latency → PD control → stepping.

        Subclasses that need a different physics pipeline should override
        ``_execute_physics`` instead of this method.
        """
        rng = kwargs.get("rng")
        kick_rng = kwargs.get("kick_rng", rng)
        push_rng = kwargs.get("push_rng", rng)
        qfrc_applied = kwargs.get("qfrc_applied", None)

        # 1. Perturbation (velocity kick + force/torque)
        data, qfrc_applied, info = self._perturbation.apply(
            data=data, info=info, qfrc_applied=qfrc_applied,
            model_nv=self.model_nv, kick_rng=kick_rng, push_rng=push_rng,
        )

        # 2. Domain-randomization forces (mass/COM/inertia/friction)
        qfrc_applied = self._physics.apply_forces(
            model=model, data=data, info=info, qfrc_applied=qfrc_applied,
        )

        # 3. Action latency → effective action (Isaac → MJX reindex)
        action_mjx = self._control.get_effective_action(action, info)

        # 4. Per-env kp/kd factors (reindex from policy to MJX order)
        kp_factor = kwargs.get("kp_factor")
        kd_factor = kwargs.get("kd_factor")
        if kp_factor is not None:
            kp_factor = kp_factor[self.isaac_to_mjx_idx]
        if kd_factor is not None:
            kd_factor = kd_factor[self.isaac_to_mjx_idx]
        motor_strength_mjx = kwargs.get("motor_strength_mjx")
        dof_damping_mjx = kwargs.get("dof_damping_mjx")

        # 5. Delegate to template method (subclasses override this)
        return self._execute_physics(
            model=model, data=data, action_mjx=action_mjx,
            kp_factor=kp_factor, kd_factor=kd_factor,
            motor_strength_mjx=motor_strength_mjx,
            dof_damping_mjx=dof_damping_mjx,
            qfrc_applied=qfrc_applied, info=info,
        )

    def _execute_physics(
        self,
        *,
        model,
        data,
        action_mjx,
        kp_factor,
        kd_factor,
        motor_strength_mjx,
        dof_damping_mjx,
        qfrc_applied,
        info,
    ):
        """Template method: run the actual physics step.

        Default: built-in MuJoCo PD actuators.  Override for custom PD,
        torque latency, etc.
        """
        from .subcomponents.control import mjx_step as _mjx_step

        motor_targets = action_mjx
        data = _mjx_step(
            model, data, motor_targets, self._control.n_substeps, qfrc_applied,
        )
        return data, jp.zeros_like(action_mjx)

    def update_contact_state(self, *, data, info, **kwargs: Any):
        """Update foot contact state. Returns dict with contact info."""
        return self._feet.update(
            data=data,
            last_contact=info["last_contact"],
            feet_air_time=info["feet_air_time"],
            feet_contact_time=info["feet_contact_time"],
        )

    def update_history(self, *, info, **kwargs: Any):
        """Push observation history. Returns updated info."""
        return self._history.push(
            info=info,
            joint_pos=kwargs["joint_pos"],
            joint_vel=kwargs["joint_vel"],
            rpy=kwargs["rpy"],
            base_ang_vel=kwargs["base_ang_vel"],
            torques=kwargs["torques"],
        )

    def update_action_state(self, *, info, **kwargs: Any):
        """Update last-action state. Returns updated info."""
        return self._history.update_action(
            info=info,
            action=kwargs["action"],
            joint_vel=kwargs["joint_vel"],
        )

    def reset_state(self, *, rng, info, done, **kwargs: Any):
        """Build reset state + reset info + reset history."""
        randomize_dof_init = kwargs.get("randomize_dof_init", False)
        reset_state = self._reset.build(rng=rng, randomize_dof_init=randomize_dof_init)
        info = self._reset.reset_info(info=info, done=done)
        info = self._history.reset_on_done(info=info, done=done)
        return info, reset_state

    def reset_history(self, *, info, done, **kwargs: Any):
        """Reset observation history buffers for done environments."""
        return self._history.reset_on_done(info=info, done=done)

    # ------------------------------------------------------------------
    # Group C: Extended facade – default implementations
    # ------------------------------------------------------------------

    @abstractmethod
    def build_initial_info(self, rng: Any, **kwargs: Any) -> dict:
        """Sample DR factors, init history, build the full info dict at reset.

        Mixed concern: contains both mechanical (DR factors, history init)
        and task-specific (command sampling) logic.
        """

    def extract_base_state(self, data: Any) -> dict:
        """Body-frame projection + joint reindexing into a dict."""
        return extract_base_state(data, self.mjx_to_isaac_idx)

    def get_odom_observation(self, data: Any) -> Any:
        """Default 29-D odom observation (rpy[:2] + dof_pos + dof_vel + ang_vel_body)."""
        quat = data.qpos[3:7]
        rpy = quat_to_rpy(quat)
        num_joints = self.isaac_to_mjx_idx.shape[0]
        joint_pos = data.qpos[7:7 + num_joints][self.isaac_to_mjx_idx]
        joint_vel = data.qvel[6:6 + num_joints][self.isaac_to_mjx_idx]
        ang_vel_world = data.qvel[3:6]
        from brax import math
        ang_vel_body = math.rotate(ang_vel_world, math.quat_inv(quat))
        return jp.concatenate([rpy[:2], joint_pos, joint_vel, ang_vel_body])

    @abstractmethod
    def maybe_resample_command(self, rng: Any, info: dict, step: Any, **kwargs: Any) -> dict:
        """Re-sample command on cadence or on reset (force=True)."""

    def finalize_reward(
        self, reward: Any, rewards: dict, *, done: Any,
        time_out: Any, info: dict,
    ) -> tuple:
        """Apply only_positive_rewards clip + termination reward."""
        if self.only_positive_rewards:
            reward = jp.clip(reward, 0.0, None)
        if self.termination_reward_scale != 0.0:
            done_float = jp.array(done).astype(jp.float32)
            time_out_float = jp.array(time_out).astype(jp.float32)
            term_reward = self.termination_reward_scale * done_float * (1.0 - time_out_float)
            reward = reward + term_reward
            rewards["termination"] = term_reward
        return reward, rewards

    def apply_perturbation(self, *, data, info, qfrc_applied, model_nv, rng) -> tuple:
        """Apply velocity kick + force/torque perturbations."""
        data, qfrc_applied, info = self._perturbation.apply(
            data=data, info=info, qfrc_applied=qfrc_applied,
            model_nv=model_nv, rng=rng,
        )
        return data, qfrc_applied, info

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def default_pose_mjx(self) -> jp.ndarray:
        """Public accessor for MJX-order default pose (used by env for qpos init)."""
        return self._control.default_pose_mjx


__all__ = ["MechanicalBase"]
