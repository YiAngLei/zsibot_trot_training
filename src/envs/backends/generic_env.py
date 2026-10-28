"""
Generic environment for MJX robot training.

Thin simulation scheduler: model loading, mjx data lifecycle, and step()
that delegates everything else to the component system.

All task-specific logic lives in the component (see src/components/).
Rendering lives in ViewerMixin (see viewer.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from brax import math
from components.factory import create_component
from components.subcomponents.kinematics import quat_to_rpy
from components.subcomponents.topology import TopologySubComponent
from envs.base import BaseEnv
from envs.info_compat import dict_to_info, info_to_dict
from envs.state import State
from .viewer import ViewerMixin


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_mj_model(config) -> tuple[mujoco.MjModel, str]:
    """Load the MuJoCo model specified by config.contact_model.
    支持多地形注入（与文档 8.2 一致）
    """
    from components.subcomponents.terrain import TerrainSubComponent
    
    contact_model = getattr(config, "contact_model", "damped")
    if getattr(config, "use_soft_contact", False) and contact_model == "default":
        contact_model = "soft"

    if contact_model == "soft":
        rel_path = "../assets/bot_model/xg/xgb_all_sphere_0128_v3_soft.xml"
        print("  [GenericEnv] Using SOFT contact model (v3_soft)")
    else:
        rel_path = "../assets/bot_model/xg/xg.xml"
        print("  [GenericEnv] Using DAMPED model (v4_damped)")

    model_path = str(_PROJECT_ROOT / rel_path)
    mj_model = mujoco.MjModel.from_xml_path(model_path)

    # ===================多地形注入=======================
    terrain_config = config.terrain
    print(f"  [GenericEnv] Generating terrain: {terrain_config.mesh_type}")
    terrain_gen = TerrainSubComponent(terrain_config)
    height_field = terrain_gen.generate_terrain(terrain_config.mesh_type)
    mj_model = terrain_gen.inject_terrain_into_model(mj_model, height_field)
    # ==================================================
    
    return mj_model, model_path


def _configure_actuators(mj_model: mujoco.MjModel, use_custom_pd: bool,
                          kp: float, kd: float) -> None:
    """Configure actuator gain/bias types and joint damping for PD control.

    In-place mutation of mj_model.
    """
    if use_custom_pd:
        for i in range(mj_model.nu):
            mj_model.actuator_gaintype[i] = 0  # FIXED
            mj_model.actuator_biastype[i] = 0   # NONE
            mj_model.actuator_gainprm[i, 0] = 0.0
            mj_model.actuator_biasprm[i, :] = 0.0
        for i in range(6, mj_model.nv):
            mj_model.dof_damping[i] = 0.0
    else:
        for i in range(mj_model.nu):
            mj_model.actuator_gaintype[i] = 1  # AFFINE
            mj_model.actuator_biastype[i] = 1   # AFFINE
            mj_model.actuator_gainprm[i, 0] = kp
            mj_model.actuator_biasprm[i, 0] = 0.0
            mj_model.actuator_biasprm[i, 1] = -kp
            mj_model.actuator_biasprm[i, 2] = 0.0
        for i in range(6, mj_model.nv):
            mj_model.dof_damping[i] = kd


def _reward_scales_from_config(config) -> dict:
    """Pre-multiply reward scales by dt (Isaac Gym convention)."""
    scales = {}
    for key, scale in config.rewards.scales.to_dict().items():
        if scale != 0:
            dt = config.sim.dt * config.control.decimation
            scales[key] = scale * dt
    return scales


class GenericEnv(ViewerMixin, BaseEnv):
    """Generic MJX robot training environment - thin simulation scheduler.

    All task logic is delegated to `self.component`; rendering is
    delegated to `ViewerMixin`.
    """

    def __init__(self, config=None, component_name: str = "velocity", **kwargs):
        if config is None:
            from cfg.xg_config import get_xg_config
            config = get_xg_config()
        self.config = config

        # 存储地形配置和高度场（用于查询）
        self._terrain_config = config.terrain
        self._terrain_height_field = None
        self._terrain_nrow = 200
        self._terrain_ncol = 200
        self._terrain_size_x = 10.0
        self._terrain_size_y = 10.0

        # 1. Load model + configure actuators
        self._mj_model, self._model_path = _load_mj_model(config)
        
        from components.subcomponents.terrain import TerrainSubComponent
        terrain_gen = TerrainSubComponent(self._terrain_config)
        height_field = terrain_gen.generate_terrain(
            self._terrain_config.mesh_type
        )
        # 转换为 JAX 数组并存储
        self._terrain_height_field = jp.array(height_field)
        self._terrain_nrow, self._terrain_ncol = self._terrain_height_field.shape
        print(f"  [GenericEnv] Terrain height field stored for query, shape: {self._terrain_height_field.shape}, range: [{height_field.min():.3f}, {height_field.max():.3f}]")
               
        use_custom_pd = getattr(config.control, "use_custom_pd", True)
        _configure_actuators(
            self._mj_model, use_custom_pd,
            config.control.stiffness, config.control.damping,
        )
        self._mj_model.opt.timestep = config.sim.dt
        self._mjx_model = mjx.put_model(self._mj_model, impl="jax")
        self._mj_data = mujoco.MjData(self._mj_model)
        self._dt = config.control.decimation * config.sim.dt

        # 2. Resolve topology (body/site/geom IDs from config-supplied names)
        self._topology = TopologySubComponent(
            mj_model=self._mj_model,
            foot_body_names=tuple(config.robot.foot_body_names),
            foot_site_names=tuple(config.robot.foot_site_names),
            nonfoot_body_names=tuple(config.robot.nonfoot_body_names),
            illegal_contact_force_threshold=getattr(
                config.termination, "illegal_contact_force_threshold", 10.0,
            ),
        )

        # 3. Build arrays from config
        default_angles = config.init_state.default_joint_angles
        default_pose = jp.array([
            default_angles.FAR_ABAD_JOINT, default_angles.FAR_HIP_JOINT, default_angles.FAR_KNEE_JOINT,
            default_angles.FBL_ABAD_JOINT, default_angles.FBL_HIP_JOINT, default_angles.FBL_KNEE_JOINT,
            default_angles.RAR_ABAD_JOINT, default_angles.RAR_HIP_JOINT, default_angles.RAR_KNEE_JOINT,
            default_angles.RBL_ABAD_JOINT, default_angles.RBL_HIP_JOINT, default_angles.RBL_KNEE_JOINT,
        ])
        isaac_to_mjx_idx = jp.arange(12)  # XG order matches Isaac Gym
        mjx_to_isaac_idx = jp.arange(12)
        joint_lower = jp.array(config.robot.joint_lower)
        joint_upper = jp.array(config.robot.joint_upper)
        kp_array = jp.ones(12) * config.control.stiffness
        kd_array = jp.ones(12) * config.control.damping
        torque_limits = jp.ones(12) * config.robot.torque_limits

        # 4. Single call to create component (no self._xxx intermediate storage)
        self.component = create_component(
            component_name,
            use_custom_pd=use_custom_pd,
            n_substeps=config.control.decimation,
            action_scale=config.control.action_scale,
            action_clip=getattr(config.control, "action_clip", 1.0),
            default_pose=default_pose,
            default_pose_mjx=default_pose,  # XG order matches MJX
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            isaac_to_mjx_idx=isaac_to_mjx_idx,
            mjx_to_isaac_idx=mjx_to_isaac_idx,
            kp_array=kp_array,
            kd_array=kd_array,
            torque_limits=torque_limits,
            feet_body_ids=self._topology.feet_body_ids,
            dt=self._dt,
            obs_dof_his_num=getattr(config.domain_rand, "obs_dof_his_num", 0),
            obs_imu_his_num=getattr(config.domain_rand, "obs_imu_his_num", 0),
            randomize_torques_latency=getattr(config.domain_rand, "randomize_torques_latency", False),
            torque_his_num=getattr(config.domain_rand, "torque_his_num", 0),
            domain_rand=config.domain_rand,
            model_nv=self._mj_model.nv,
            config=config,
            # Perturbation
            push_robots=getattr(config.domain_rand, "push_robots", False),
            push_interval=int(getattr(config.domain_rand, "push_interval_s", 5.0) / self._dt),
            max_push_vel_xy=getattr(config.domain_rand, "max_push_vel_xy", 0.5),
            push_robots_fm=getattr(config.domain_rand, "push_robots_FM", False),
            push_fm_interval=int(getattr(config.domain_rand, "push_FM_interval_s", 15.0) / self._dt),
            push_fm_duration=int(getattr(config.domain_rand, "push_FM_duration_s", 1.0) / self._dt),
            max_push_force_fm=getattr(config.domain_rand, "max_push_force_FM", 50.0),
            max_push_torque_fm=getattr(config.domain_rand, "max_push_torque_FM", 5.0),
            # Init noise
            init_jitter_height=config.robot.init_jitter_height,
            init_dof_noise=config.robot.init_dof_noise,
            init_dof_vel_noise=config.robot.init_dof_vel_noise,
            init_base_vel_noise=config.robot.init_base_vel_noise,
            # Reward termination
            only_positive_rewards=config.rewards.only_positive_rewards,
            termination_reward_scale=_reward_scales_from_config(config).get("termination", 0.0),
        )

        # 5. Validate
        self._reward_scales = {
            k: v for k, v in _reward_scales_from_config(config).items() if k != "termination"
        }
        self.component.validate_reward_scales(self._reward_scales)

        # 6. Viewer init (state only, no launch)
        self._enable_viewer = False
        self._viewer = None

    # ⭐ ========== 地形高度查询方法（JAX兼容版本）==========
    def _query_terrain_height(self, x, y):
        """
        查询地形高度（双线性插值）- 纯 JAX 版本，可被 JIT 编译
        
        Args:
            x: 世界坐标系 x 坐标 (范围 -10 ~ 10)
            y: 世界坐标系 y 坐标 (范围 -10 ~ 10)
        
        Returns:
            该位置的地形高度
        """
        if self._terrain_height_field is None:
            return 0.0
        
        nrow, ncol = self._terrain_nrow, self._terrain_ncol
        size_x, size_y = self._terrain_size_x, self._terrain_size_y
        
        # 将世界坐标映射到高度场索引
        col_idx = (x + size_x) / (2 * size_x) * (ncol - 1)
        row_idx = (y + size_y) / (2 * size_y) * (nrow - 1)
        
        # 边界裁剪
        col_idx = jp.clip(col_idx, 0, ncol - 1)
        row_idx = jp.clip(row_idx, 0, nrow - 1)
        
        # 获取四个角点索引
        col_0 = jp.floor(col_idx).astype(jp.int32)
        col_1 = jp.minimum(col_0 + 1, ncol - 1)
        row_0 = jp.floor(row_idx).astype(jp.int32)
        row_1 = jp.minimum(row_0 + 1, nrow - 1)
        
        # ⭐ 使用展平索引访问（JAX兼容）
        flat_idx_00 = row_0 * ncol + col_0
        flat_idx_01 = row_0 * ncol + col_1
        flat_idx_10 = row_1 * ncol + col_0
        flat_idx_11 = row_1 * ncol + col_1
        
        h_flat = self._terrain_height_field.ravel()
        h00 = h_flat[flat_idx_00]
        h01 = h_flat[flat_idx_01]
        h10 = h_flat[flat_idx_10]
        h11 = h_flat[flat_idx_11]
        
        # 双线性插值权重
        wx = col_idx - col_0
        wy = row_idx - row_0
        
        # 先沿 x 方向插值
        h_top = h00 * (1 - wx) + h01 * wx
        h_bottom = h10 * (1 - wx) + h11 * wx
        # 再沿 y 方向插值
        h = h_top * (1 - wy) + h_bottom * wy
        
        return h

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_size(self) -> Dict[str, int]:
        return {
            "state": getattr(self.config.robot, "num_actor_obs", 48),
            "privileged": getattr(self.config.robot, "num_privileged_obs", 259),
        }

    @property
    def action_size(self) -> int:
        return getattr(self.config.env, "num_actions", 12)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def unwrapped(self) -> "GenericEnv":
        return self

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        """Reset env: build qpos/qvel, mjx.make_data, sample info, compute obs."""
        randomize_dof_init = bool(
            self.config.domain_rand and getattr(self.config.domain_rand, "randomize_dof_init", False)
        )

        # Match the original env's RNG consumption order exactly.
        rng, cmd_rng, pos_rng, vel_rng, height_rng = jax.random.split(rng, 5)
        rng, motor_rng, kp_rng, kd_rng, latency_rng = jax.random.split(rng, 5)
        rng, dof_damping_rng, friction_rng, inertia_rng, restitution_rng = jax.random.split(rng, 5)
        rng, mass_rng, com_offset_rng = jax.random.split(rng, 3)

        # 1. Build initial physics state (env-level concern)
        qpos, qvel = self._build_initial_state(
            pos_rng, vel_rng, height_rng, randomize_dof_init
        )

        data = mjx.make_data(self._mjx_model, impl="jax")
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._mjx_model, data)

        # 2. Build info dict (delegated to component)
        info_dict = self.component.build_initial_info(
            rng,
            cmd_rng=cmd_rng,
            motor_rng=motor_rng,
            kp_rng=kp_rng,
            kd_rng=kd_rng,
            damp_rng=dof_damping_rng,
            fric_rng=friction_rng,
            inrt_rng=inertia_rng,
            res_rng=restitution_rng,
            mass_rng=mass_rng,
            com_rng=com_offset_rng,
            action_lat_rng=latency_rng,
        )
        info_dict["odom_ground_truth"] = data.qvel[:3]

        # ⭐ 将地形查询函数放入 info
        if self._terrain_height_field is not None:
            info_dict["terrain_query"] = self._query_terrain_height

        # 3. Compute initial observation
        obs = self.component.build_observation(
            data=data, info=info_dict, action=jp.zeros(12),
            default_pose=self.component.default_pose_mjx,
            mjx_to_policy_idx=self.component.mjx_to_isaac_idx,
            feet_body_ids=self._topology.feet_body_ids,
            randomize_obs_latency=self._randomize_obs_latency(),
            obs_imu_his_num=self._obs_imu_his(),
            obs_dof_his_num=self._obs_dof_his(),
            add_noise=getattr(self.config.noise, "add_noise", False),
            noise_config=self.config.noise,
        )

        # Convert dict → EnvInfo for State (typed schema)
        info = dict_to_info(info_dict)

        return State(
            data=data, obs=obs, reward=jp.array(0.0), done=jp.array(0.0),
            metrics=self._default_metrics(), info=info,
        )

    def _build_initial_state(self, pos_rng, vel_rng, height_rng, randomize_dof_init: bool) -> tuple:
        """Build initial qpos and qvel from pre-split RNGs."""
        cfg = self.config
        robot = cfg.robot

        init_pos = jp.array(cfg.init_state.pos)
        init_rot = jp.array(cfg.init_state.rot)
        random_height = jax.random.uniform(height_rng, (), minval=0.0, maxval=robot.init_jitter_height)
        init_pos = init_pos.at[2].set(init_pos[2] + random_height)

        if randomize_dof_init:
            dof_noise = jax.random.uniform(
                pos_rng, (12,),
                minval=-robot.init_dof_noise, maxval=robot.init_dof_noise,
            )
            joint_pos_mjx = self.component.default_pose_mjx + dof_noise
            joint_vel_noise = jax.random.uniform(
                vel_rng, (12,),
                minval=-robot.init_dof_vel_noise, maxval=robot.init_dof_vel_noise,
            )
        else:
            joint_pos_mjx = self.component.default_pose_mjx
            joint_vel_noise = jp.zeros(12)

        qpos = jp.concatenate([init_pos, init_rot, joint_pos_mjx])
        base_vel_noise = jax.random.uniform(
            vel_rng, (6,),
            minval=-robot.init_base_vel_noise, maxval=robot.init_base_vel_noise,
        )
        qvel = jp.zeros(self._mj_model.nv)
        qvel = qvel.at[:6].set(base_vel_noise)
        qvel = qvel.at[6:18].set(joint_vel_noise)
        return qpos, qvel

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, state: State, action: jax.Array, odom_data: jax.Array = None,
             clip_action: bool = True) -> State:
        """One env step. All heavy logic is in self.component."""
        if clip_action:
            action = jp.clip(action, -self.component.action_clip, self.component.action_clip)
        else:
            # Match the legacy xg_env_simple training path: actions are still
            # bounded, but only by the wide Isaac-style training clip.
            action = jp.clip(action, -100.0, 100.0)
        data = state.data
        # Convert EnvInfo → dict for internal use (sub-components expect dict)
        info = info_to_dict(state.info)
        info["odom_data"] = odom_data

        # ⭐ 确保地形查询函数在 info 中
        if self._terrain_height_field is not None and "terrain_query" not in info:
            info["terrain_query"] = self._query_terrain_height

        # Match the original env's step-level RNG split order.
        rng, cmd_rng, kick_rng, push_rng = jax.random.split(info["rng"], 4)
        info["rng"] = rng

        # 1. Physics: perturbation + DR forces + PD control + stepping
        data, torques = self.component.step_physics(
            model=self._mjx_model, data=data, action=action, info=info,
            rng=rng,
            kick_rng=kick_rng,
            push_rng=push_rng,
            kp_factor=info.get("kp_factor"),
            kd_factor=info.get("kd_factor"),
            motor_strength_mjx=info.get("motor_strength")[self.component.isaac_to_mjx_idx]
                if "motor_strength" in info else None,
            dof_damping_mjx=info.get("dof_damping")[self.component.isaac_to_mjx_idx]
                if "dof_damping" in info else None,
        )

        # 2. Body-frame kinematics (single helper call)
        base = self.component.extract_base_state(data)

        # 3. Update proprioceptive history (uses rpy + torques)
        # Match the original env: history stores roll/pitch and zeros yaw.
        rpy = quat_to_rpy(base["base_quat"]).at[2].set(0.0)
        info = self.component.update_history(
            info=info, joint_pos=base["joint_pos"], joint_vel=base["joint_vel"],
            rpy=rpy, base_ang_vel=base["base_ang_vel"], torques=torques,
        )

        # 4. Step counter + odom bookkeeping
        info["rng"] = rng
        info["step"] = info["step"] + 1
        info["common_step_counter"] = info["common_step_counter"] + 1
        if odom_data is None:
            info["odom_data"] = base["base_lin_vel"]
        info["odom_ground_truth"] = base["base_lin_vel"]

        # 4.5 Heading command update (convert heading_target to angular velocity)
        if getattr(self.config.commands, "heading_command", False):
            base_quat = base["base_quat"]
            forward_vec = jp.array([1.0, 0.0, 0.0])
            forward = math.rotate(forward_vec, base_quat)
            heading = jp.arctan2(forward[1], forward[0])
            heading_target = info.get("heading_target", jp.array(0.0))
            heading_error = heading_target - heading
            heading_error = jp.arctan2(jp.sin(heading_error), jp.cos(heading_error))
            heading_control_stiffness = getattr(
                self.config.commands, "heading_control_stiffness", 0.5
            )
            new_cmd_yaw = jp.clip(heading_control_stiffness * heading_error, -1.0, 1.0)
            command = info["command"]
            info["command"] = command.at[2].set(new_cmd_yaw)

        # 5. Termination
        terminations = self.component.compute_terminations(
            data=data, projected_gravity=base["projected_gravity"],
            nonfoot_body_ids=self._topology.nonfoot_body_ids,
            illegal_contact_force_threshold=self._topology.illegal_contact_force_threshold,
            fell_over_limit_angle=getattr(
                self.config.termination, "fell_over_limit_angle", 1.22,
            ),
            step=info["step"],
            episode_length=self.config.env.episode_length,
        )
        done = terminations["done"]
        time_out = terminations["time_out"]
        info = self.component.write_termination_info(info=info, terminations=terminations)

        # 6. Foot contact
        contact = self.component.update_contact_state(data=data, info=info)
        info["feet_air_time"] = contact["feet_air_time"]
        info["feet_contact_time"] = contact["feet_contact_time"]
        info["last_contact"] = contact["contact"]

        # 7. Rewards
        rewards = self.component.compute_rewards(
            reward_scales=self._reward_scales,
            reward_config=self.config.rewards,
            context=self.component.build_reward_context(
                command=info["command"], action=action,
                last_action=info["last_action"], joint_pos=base["joint_pos"],
                default_pose=self.component.default_pose_mjx,
                soft_joint_lower=(self.component._control.joint_lower
                    + self.component._control.joint_upper) * 0.5
                    - 0.5 * (self.component._control.joint_upper
                        - self.component._control.joint_lower
                    ) * getattr(self.config.rewards, "soft_dof_pos_limit", 0.9),
                soft_joint_upper=(self.component._control.joint_lower
                    + self.component._control.joint_upper) * 0.5
                    + 0.5 * (self.component._control.joint_upper
                        - self.component._control.joint_lower
                    ) * getattr(self.config.rewards, "soft_dof_pos_limit", 0.9),
                base_lin_vel=base["base_lin_vel"],
                base_ang_vel=base["base_ang_vel"],
                projected_gravity=base["projected_gravity"],
                data=data, feet_body_ids=self._topology.feet_body_ids,
                feet_air_time=info.get("feet_air_time", jp.zeros(4)),
                first_contact=contact["first_contact"],
            ),
        )
        # Stateful rewards (e.g. foot_swing_height with peak-height tracking)
        stateful_rewards, stateful_updates = self.component.compute_stateful_rewards(
            reward_scales=self._reward_scales, info=info, data=data,
            feet_body_ids=self._topology.feet_body_ids,
            contact_filt=contact["contact_filt"],
            first_contact=contact["first_contact"],
            reward_config=self.config.rewards,
        )
        for rname, rval in stateful_rewards.items():
            rewards[rname] = rval * self._reward_scales[rname]
        # Stateful terms are computed separately, but reward accumulation must
        # follow the task registry order to match the original floating-point
        # reduction exactly.
        rewards = {
            name: rewards[name]
            for name in self.component.reward_terms
            if name in rewards
        }
        info.update(stateful_updates)
        info["rewards"] = rewards

        # 8. Last action update (after rewards, before reset)
        info = self.component.update_action_state(
            info=info, action=action, joint_vel=base["joint_vel"],
        )

        # 9. Reward total + termination reward
        reward = sum(rewards.values()) if rewards else jp.array(0.0)
        reward, rewards = self.component.finalize_reward(
            reward, rewards, done=done, time_out=time_out, info=info,
        )

        # 10. Observation
        obs = self.component.build_observation(
            data=data, info=info, action=action,
            default_pose=self.component.default_pose_mjx, mjx_to_policy_idx=self.component.mjx_to_isaac_idx,
            feet_body_ids=self._topology.feet_body_ids,
            randomize_obs_latency=self._randomize_obs_latency(),
            obs_imu_his_num=self._obs_imu_his(),
            obs_dof_his_num=self._obs_dof_his(),
            add_noise=getattr(self.config.noise, "add_noise", False),
            noise_config=self.config.noise,
        )

        # 11. Command resample (cadence)
        info = self.component.maybe_resample_command(cmd_rng, info, info["step"])

        # 12. Reset on done
        _randomize_dof_init = bool(
            self.config.domain_rand
            and getattr(self.config.domain_rand, "randomize_dof_init", False)
        )
        rng, reset_rng = jax.random.split(info["rng"])
        info["rng"] = rng
        info, reset_state = self.component.reset_state(
            rng=reset_rng, info=info, done=done,
            randomize_dof_init=_randomize_dof_init,
        )
        info["last_base_lin_vel"] = jp.where(~done, base["base_lin_vel"], info["last_base_lin_vel"])

        # 13. Apply reset to physics state
        data = data.replace(
            qpos=jp.where(done, reset_state["qpos"], data.qpos),
            qvel=jp.where(done, reset_state["qvel"], data.qvel),
        )
        data = mjx.forward(self._mjx_model, data)
        info["torque_his"] = jp.where(
            done, reset_state["torque_his"], info["torque_his"]
        )

        # 14. Resample command on reset + recompute obs (only for done envs)
        _, reset_cmd_rng = jax.random.split(reset_state["rng"], 2)
        info = self.component.maybe_resample_command(
            reset_cmd_rng, info, info["step"], force=True, done=done,
        )
        obs = self.component.build_observation(
            data=data, info=info, action=jp.where(done, jp.zeros(12), action),
            default_pose=self.component.default_pose_mjx, mjx_to_policy_idx=self.component.mjx_to_isaac_idx,
            feet_body_ids=self._topology.feet_body_ids,
            randomize_obs_latency=self._randomize_obs_latency(),
            obs_imu_his_num=self._obs_imu_his(),
            obs_dof_his_num=self._obs_dof_his(),
            add_noise=getattr(self.config.noise, "add_noise", False),
            noise_config=self.config.noise,
        )

        # 15. Metrics
        metrics = state.metrics.copy() if hasattr(state, "metrics") else self._default_metrics()
        metrics["total_dist"] = jp.linalg.norm(base["base_pos"][:2])
        metrics["reward_ori"] = reward
        metrics["reward_scaled"] = reward
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = v

        # Convert dict → EnvInfo for State (typed schema)
        info = dict_to_info(info)

        return State(
            data=data, obs=obs, reward=reward,
            done=done.astype(jp.float32), metrics=metrics, info=info,
        )

    # ------------------------------------------------------------------
    # Small accessors (avoid long attribute chains in step())
    # ------------------------------------------------------------------

    def _randomize_obs_latency(self) -> bool:
        return bool(
            self.config.domain_rand
            and getattr(self.config.domain_rand, "randomize_obs_latency", False)
        )

    def _obs_dof_his(self) -> int:
        return getattr(self.config.domain_rand, "obs_dof_his_num", 0) \
            if self.config.domain_rand else 0

    def _obs_imu_his(self) -> int:
        return getattr(self.config.domain_rand, "obs_imu_his_num", 0) \
            if self.config.domain_rand else 0

    @staticmethod
    def _default_metrics() -> dict:
        return {"total_dist": jp.array(0.0), "reward_ori": jp.array(0.0), "reward_scaled": jp.array(0.0)}

    # ------------------------------------------------------------------
    # Odom (delegated to component)
    # ------------------------------------------------------------------

    def get_odom_observations(self, state: State) -> jax.Array:
        return self.component.get_odom_observation(state.data)