"""Velocity-tracking locomotion component."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp

from ...mechanical_base import MechanicalBase
from ...subcomponents.control import ControlSubComponent
from ...subcomponents.feet import FeetSubComponent
from ...subcomponents.history import HistorySubComponent
from ...subcomponents.perturbation import PerturbationSubComponent
from ...subcomponents.physics import PhysicsSubComponent
from ...subcomponents.reset import ResetSubComponent

from .commands import sample_velocity_command
from .observations import (
    VELOCITY_ACTOR_SCHEMA,
    VELOCITY_PRIVILEGED_SCHEMA,
    build_velocity_observation,
)
from .rewards import (
    VELOCITY_REWARD_TERMS,
    VelocityRewardContext,
    compute_extracted_velocity_rewards,
    compute_foot_swing_height_reward,
)
from .terminations import (
    VELOCITY_TERMINATION_TERMS,
    compute_velocity_terminations,
    write_termination_info,
)

NUM_JOINTS = 12  # XG quadruped: 4 legs × 3 joints


class VelocityComponent(MechanicalBase):
    """Velocity-tracking task component.

    Inherits mechanical orchestration from MechanicalBase and only
    implements task-specific logic (commands, observations, rewards,
    terminations) plus custom PD physics stepping.
    """

    name = "velocity"
    reward_terms = VELOCITY_REWARD_TERMS
    termination_terms = VELOCITY_TERMINATION_TERMS
    required_rewards = (
        "tracking_lin_vel",
        "tracking_ang_vel",
        "upright",
        "pose",
        "dof_pos_limits",
        "action_rate",
        "feet_air_time",
        "foot_clearance",
        "slip",
        "soft_landing",
        "lateral_vel",
        "foot_swing_height",
        "yaw_rate_error",
    )
    required_terminations = ("fell_over", "illegal_contact", "nan", "timeout")

    def __init__(
        self,
        *,
        use_custom_pd: bool,
        n_substeps: int,
        action_scale: float,
        default_pose_mjx: Any,
        default_pose: Any,
        joint_lower: Any,
        joint_upper: Any,
        isaac_to_mjx_idx: Any,
        mjx_to_isaac_idx: Any,
        kp_array: Any,
        kd_array: Any,
        torque_limits: Any,
        action_clip: float = 100.0,
        randomize_torques_latency: bool = False,
        torque_his_num: int = 0,
        feet_body_ids: Any | None = None,
        dt: float = 0.01,
        obs_dof_his_num: int = 0,
        obs_imu_his_num: int = 0,
        domain_rand: Any | None = None,
        model_nv: int = 18,
        config: Any | None = None,
        # Perturbation
        push_robots: bool = False,
        push_interval: int = 500,
        max_push_vel_xy: float = 0.5,
        push_robots_fm: bool = False,
        push_fm_interval: int = 500,
        push_fm_duration: int = 200,
        max_push_force_fm: float = 50.0,
        max_push_torque_fm: float = 10.0,
        # Init noise
        init_jitter_height: float = 0.2,
        init_dof_noise: float = 0.1,
        init_dof_vel_noise: float = 0.01,
        init_base_vel_noise: float = 0.5,
        # Odom
        odom_obs_size: int = 29,
        # Termination reward
        only_positive_rewards: bool = False,
        termination_reward_scale: float = 0.0,
    ):
        # Stash raw configs for later
        self.config = config
        self.action_clip = action_clip
        self.domain_rand = domain_rand
        self.obs_dof_his_num = obs_dof_his_num
        self.obs_imu_his_num = obs_imu_his_num
        self.torque_his_num = torque_his_num
        self.model_nv = model_nv
        self.isaac_to_mjx_idx = isaac_to_mjx_idx
        self.mjx_to_isaac_idx = mjx_to_isaac_idx
        self.dt = dt
        self.use_custom_pd = use_custom_pd
        self.odom_obs_size = odom_obs_size
        self.only_positive_rewards = only_positive_rewards
        self.termination_reward_scale = termination_reward_scale
        # Init noise scales
        self.init_jitter_height = init_jitter_height
        self.init_dof_noise = init_dof_noise
        self.init_dof_vel_noise = init_dof_vel_noise
        self.init_base_vel_noise = init_base_vel_noise

        # Compose sub-components (private)
        self._control = ControlSubComponent(
            use_custom_pd=use_custom_pd,
            n_substeps=n_substeps,
            randomize_torques_latency=randomize_torques_latency,
            torque_his_num=torque_his_num,
            action_scale=action_scale,
            default_pose_mjx=default_pose_mjx,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            isaac_to_mjx_idx=isaac_to_mjx_idx,
            mjx_to_isaac_idx=mjx_to_isaac_idx,
            kp_array=kp_array,
            kd_array=kd_array,
            torque_limits=torque_limits,
        )
        self._feet = FeetSubComponent(
            feet_body_ids=feet_body_ids if feet_body_ids is not None else jp.zeros(4, dtype=jp.int32),
            dt=dt,
        )
        self._history = HistorySubComponent(
            default_pose=default_pose,
            obs_dof_his_num=obs_dof_his_num,
            obs_imu_his_num=obs_imu_his_num,
            use_custom_pd=use_custom_pd,
        )
        self._physics = PhysicsSubComponent(domain_rand=domain_rand)
        self._reset = ResetSubComponent(
            model_nv=model_nv,
            default_pose=default_pose,
            default_pose_mjx=default_pose_mjx,
            isaac_to_mjx_idx=isaac_to_mjx_idx,
            mjx_to_isaac_idx=mjx_to_isaac_idx,
            torque_his_num=torque_his_num,
            use_custom_pd=use_custom_pd,
            kp_array=kp_array,
            kd_array=kd_array,
            torque_limits=torque_limits,
            push_robots_fm=push_robots_fm,
            obs_dof_his_num=obs_dof_his_num,
            obs_imu_his_num=obs_imu_his_num,
            config=config,
        )
        self._perturbation = PerturbationSubComponent(
            push_robots=push_robots,
            push_interval=push_interval,
            max_push_vel_xy=max_push_vel_xy,
            push_robots_fm=push_robots_fm,
            push_fm_interval=push_fm_interval,
            push_fm_duration=push_fm_duration,
            max_push_force_fm=max_push_force_fm,
            max_push_torque_fm=max_push_torque_fm,
        )

    # ------------------------------------------------------------------
    # Task-specific methods (delegate to task modules)
    # ------------------------------------------------------------------

    def sample_command(self, **kwargs: Any):
        return sample_velocity_command(**kwargs)

    def build_observation(self, **kwargs: Any):
        return build_velocity_observation(**kwargs)

    def build_reward_context(self, **kwargs: Any) -> VelocityRewardContext:
        return VelocityRewardContext(**kwargs)

    def compute_rewards(self, **kwargs: Any):
        return compute_extracted_velocity_rewards(**kwargs)

    def compute_foot_swing_height_reward(self, **kwargs: Any):
        return compute_foot_swing_height_reward(**kwargs)

    def compute_terminations(self, **kwargs: Any):
        return compute_velocity_terminations(**kwargs)

    def write_termination_info(self, **kwargs: Any):
        return write_termination_info(**kwargs)

    def compute_stateful_rewards(self, *, reward_scales, info, data, **kwargs: Any) -> tuple[dict, dict]:
        """Velocity task: foot_swing_height reward with peak-height tracking."""
        if "foot_swing_height" not in reward_scales:
            return {}, {}
        fsh, peak = compute_foot_swing_height_reward(
            data=data,
            feet_body_ids=kwargs.get("feet_body_ids"),
            contact_filt=kwargs.get("contact_filt"),
            first_contact=kwargs.get("first_contact"),
            peak_heights=info.get("peak_heights", jp.zeros(4)),
            command=info.get("command"),
            reward_config=kwargs.get("reward_config"),
        )
        return {"foot_swing_height": fsh}, {"peak_heights": peak}

    # ------------------------------------------------------------------
    # Physics override: custom PD + torque latency
    # ------------------------------------------------------------------

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
        """Override: custom PD branch with optional torque latency."""
        from ...subcomponents.control import (
            compute_pd_torques,
            mjx_step_custom_pd,
            mjx_step_with_torques,
        )

        target_pos_mjx = self._control.default_pose_mjx + action_mjx * self._control.action_scale
        target_pos_mjx = jp.clip(
            target_pos_mjx,
            self._control.joint_lower[self.isaac_to_mjx_idx],
            self._control.joint_upper[self.isaac_to_mjx_idx],
        )
        effective_kp = self._control.kp_array * (kp_factor if kp_factor is not None else 1.0)
        effective_kd = self._control.kd_array * (kd_factor if kd_factor is not None else 1.0)

        if self._control.randomize_torques_latency:
            new_torques_mjx = compute_pd_torques(
                data, target_pos_mjx, effective_kp, effective_kd,
                self._control.torque_limits, motor_strength_mjx, dof_damping_mjx,
            )
            # Update history with fresh torques
            info = self._control.push_torque_history(info, new_torques_mjx)
            # Select delayed or fresh torques
            applied, fresh = self._control.select_applied_torques(
                info["rng"], info, new_torques_mjx, info["step"],
            )
            data = mjx_step_with_torques(
                model, data, applied, self._control.n_substeps, qfrc_applied,
            )
            return data, new_torques_mjx
        else:
            data, torques = mjx_step_custom_pd(
                model, data, target_pos_mjx, effective_kp, effective_kd,
                self._control.torque_limits, self._control.n_substeps, qfrc_applied,
                motor_strength_mjx, dof_damping_mjx,
            )
            return data, torques

    # ------------------------------------------------------------------
    # Mixed-concern overrides (task + mechanical)
    # ------------------------------------------------------------------

    def build_initial_info(self, rng: jax.Array, **kwargs: Any) -> dict:
        """Sample DR factors, init history, build the full info dict at reset."""
        info: dict = {}
        cmd_rng = kwargs.get("cmd_rng")
        motor_rng = kwargs.get("motor_rng")
        kp_rng = kwargs.get("kp_rng")
        kd_rng = kwargs.get("kd_rng")
        damp_rng = kwargs.get("damp_rng")
        fric_rng = kwargs.get("fric_rng")
        inrt_rng = kwargs.get("inrt_rng")
        mass_rng = kwargs.get("mass_rng")
        com_rng = kwargs.get("com_rng")
        res_rng = kwargs.get("res_rng")
        action_lat_rng = kwargs.get("action_lat_rng")
        if any(x is None for x in (
            cmd_rng, motor_rng, kp_rng, kd_rng, damp_rng, fric_rng,
            inrt_rng, mass_rng, com_rng, res_rng, action_lat_rng,
        )):
            rng, cmd_rng, motor_rng, kp_rng, kd_rng, damp_rng, fric_rng, \
                inrt_rng, mass_rng, com_rng, res_rng, _feet_fric_rng, action_lat_rng = \
                jax.random.split(rng, 13)

        # Command sample (task-specific)
        heading_command = getattr(self.config.commands, "heading_command", False) \
            if self.config is not None else False
        command, heading_target = self.sample_command(
            rng=cmd_rng,
            command_config=self.config.commands if self.config is not None else None,
            heading_command=heading_command,
        )
        info["command"] = command
        info["heading_target"] = heading_target

        # DR factors
        if self.domain_rand and getattr(self.domain_rand, "randomize_motor_strength", False):
            motor_range = self.domain_rand.motor_strength_range
            info["motor_strength"] = jax.random.uniform(
                motor_rng, (NUM_JOINTS,), minval=motor_range[0], maxval=motor_range[1],
            )
        else:
            info["motor_strength"] = jp.ones(NUM_JOINTS)

        if self.domain_rand and getattr(self.domain_rand, "randomize_Kp_factor", False):
            kp_range = self.domain_rand.Kp_factor_range
            info["kp_factor"] = jax.random.uniform(
                kp_rng, (NUM_JOINTS,), minval=kp_range[0], maxval=kp_range[1],
            )
        else:
            info["kp_factor"] = jp.ones(NUM_JOINTS)

        if self.domain_rand and getattr(self.domain_rand, "randomize_Kd_factor", False):
            kd_range = self.domain_rand.Kd_factor_range
            info["kd_factor"] = jax.random.uniform(
                kd_rng, (NUM_JOINTS,), minval=kd_range[0], maxval=kd_range[1],
            )
        else:
            info["kd_factor"] = jp.ones(NUM_JOINTS)

        if self.domain_rand and getattr(self.domain_rand, "randomize_dof_damping", False):
            damping_range = getattr(self.domain_rand, "dof_damping_range", (0.0, 0.5))
            info["dof_damping"] = jax.random.uniform(
                damp_rng,
                (NUM_JOINTS,),
                minval=damping_range[0],
                maxval=damping_range[1],
            )
        else:
            info["dof_damping"] = jp.zeros(NUM_JOINTS)

        if self.domain_rand and getattr(self.domain_rand, "randomize_friction", False):
            friction_range = self.domain_rand.friction_range
            info["friction_coeff"] = jax.random.uniform(
                fric_rng, (), minval=friction_range[0], maxval=friction_range[1],
            )
        else:
            info["friction_coeff"] = jp.array(1.0)

        if self.domain_rand and getattr(self.domain_rand, "randomize_inertia", False):
            inertia_range = self.domain_rand.inertia_factor_range
            info["inertia_factors"] = jax.random.uniform(
                inrt_rng, (1, 3), minval=inertia_range[0], maxval=inertia_range[1],
            )
        else:
            info["inertia_factors"] = jp.ones((1, 3))

        if self.domain_rand and getattr(self.domain_rand, "randomize_mass", False):
            mass_range = self.domain_rand.mass_factor_range
            info["mass_factors"] = jax.random.uniform(
                mass_rng, (1,), minval=mass_range[0], maxval=mass_range[1],
            )
        else:
            info["mass_factors"] = jp.ones((1,))

        if self.domain_rand and getattr(self.domain_rand, "randomize_com_offset", False):
            com_range = self.domain_rand.com_offset_range
            if isinstance(com_range[0], (tuple, list)):
                com_offset = jp.array([
                    jax.random.uniform(
                        com_rng, (), minval=com_range[0][0], maxval=com_range[0][1]
                    ),
                    jax.random.uniform(
                        jax.random.split(com_rng)[0], (), minval=com_range[1][0], maxval=com_range[1][1]
                    ),
                    jax.random.uniform(
                        jax.random.split(com_rng)[1], (), minval=com_range[2][0], maxval=com_range[2][1]
                    ),
                ])
            else:
                com_offset = jax.random.uniform(
                    com_rng, (3,), minval=com_range[0], maxval=com_range[1],
                )
            info["com_offset"] = com_offset
        else:
            info["com_offset"] = jp.zeros(3)

        if self.domain_rand and getattr(self.domain_rand, "randomize_restitution", False):
            res_range = self.domain_rand.restitution_range
            info["restitution_coeff"] = jax.random.uniform(
                res_rng, (), minval=res_range[0], maxval=res_range[1],
            )
        else:
            info["restitution_coeff"] = jp.array(0.0)

        info["feet_friction"] = info["friction_coeff"]

        if self.domain_rand and getattr(self.domain_rand, "randomize_action_latency", False):
            latency_range = self.domain_rand.action_latency_range
            latency_min = int((latency_range[0] + 1e-8) / self.config.sim.dt)
            latency_max = int((latency_range[1] - 1e-8) / self.config.sim.dt) + 1
            info["action_latency"] = jax.random.randint(
                action_lat_rng,
                (),
                latency_min,
                latency_max,
            )
        else:
            info["action_latency"] = jp.array(0, dtype=jp.int32)

        # History buffers
        info["dof_pos_history"] = jp.ones((NUM_JOINTS, self.obs_dof_his_num + 1)) * self._control.default_pose_mjx[:, None]
        info["dof_vel_history"] = jp.zeros((NUM_JOINTS, self.obs_dof_his_num + 1))
        info["rpy_history"] = jp.zeros((3, self.obs_imu_his_num + 1))
        info["ang_lin_history"] = jp.zeros((3, self.obs_imu_his_num + 1))
        if self.use_custom_pd:
            info["torque_his"] = jp.zeros((NUM_JOINTS, self.torque_his_num + 1))
        else:
            info["torque_his"] = jp.zeros((NUM_JOINTS, max(1, self.torque_his_num + 1)))

        # Step counters + base state placeholders
        info["step"] = jp.array(0, dtype=jp.int32)
        info["common_step_counter"] = jp.array(0, dtype=jp.int32)
        info["rng"] = rng
        info["last_action"] = jp.zeros(NUM_JOINTS)
        info["last_last_action"] = jp.zeros(NUM_JOINTS)
        info["last_dof_vel"] = jp.zeros(NUM_JOINTS)
        info["last_last_dof_vel"] = jp.zeros(NUM_JOINTS)
        info["last_base_lin_vel"] = jp.zeros(3)

        # Contact + foot times
        info["last_contact"] = jp.zeros(4, dtype=bool)
        info["feet_air_time"] = jp.zeros(4)
        info["feet_contact_time"] = jp.zeros(4)
        info["peak_heights"] = jp.zeros(4)

        # Timeouts + flags
        info["unflip_time_out"] = jp.array(0, dtype=jp.int32)
        info["speed_unfollow_time_out"] = jp.array(0, dtype=jp.int32)
        info["low_height_time_out"] = jp.array(0, dtype=jp.int32)
        info["collision_time_out"] = jp.array(0, dtype=jp.int32)
        info["cumulative_speed"] = jp.array(0.0)
        info["cumulative_cmd_speed"] = jp.array(0.0)
        info["ever_flipped"] = jp.array(False)

        # Termination flags (initial)
        info["done_fell_over"] = jp.array(False)
        info["done_illegal_contact"] = jp.array(False)
        info["done_nan"] = jp.array(False)
        info["time_out"] = jp.array(False)
        info["done_unflip"] = jp.array(False)

        # Rewards (initial empty dict; populated in step())
        info["rewards"] = {}

        # Odom
        info["odom_data"] = jp.zeros(3)

        # Push-FM state
        info["push_FM_active"] = jp.array(False)
        info["push_FM_step_count"] = jp.array(0, dtype=jp.int32)
        info["push_force"] = jp.zeros(3)
        info["push_torque"] = jp.zeros(3)

        return info

    def maybe_resample_command(
        self, rng: jax.Array, info: dict, step: Any, **kwargs: Any,
    ) -> dict:
        """Re-sample command on cadence or on reset (force=True)."""
        force = kwargs.get("force", False)
        done = kwargs.get("done", None)
        resampling_time = getattr(
            self.config.commands, "resampling_time", 8.0,
        ) if self.config is not None else 8.0
        resample_interval = int(resampling_time / self.dt)
        should_resample = (step % resample_interval) == 0
        should_resample = should_resample | force
        if done is not None:
            should_resample = should_resample & done

        heading_command = getattr(self.config.commands, "heading_command", False) \
            if self.config is not None else False
        new_cmd, new_heading = self.sample_command(
            rng=rng,
            command_config=self.config.commands if self.config is not None else None,
            heading_command=heading_command,
        )
        info["command"] = jp.where(should_resample, new_cmd, info["command"])
        info["heading_target"] = jp.where(should_resample, new_heading, info["heading_target"])
        return info


__all__ = ["VelocityComponent"]
