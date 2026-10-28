"""Reset-state construction sub-component."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp


class ResetSubComponent:
    """Reset state building and info reset sub-component."""

    def __init__(self, *, model_nv: int, default_pose: jax.Array,
                 default_pose_mjx: jax.Array,
                 isaac_to_mjx_idx: jax.Array,
                 mjx_to_isaac_idx: jax.Array,
                 torque_his_num: int, use_custom_pd: bool,
                 kp_array: jax.Array, kd_array: jax.Array,
                 torque_limits: jax.Array,
                 push_robots_fm: bool = False,
                 obs_dof_his_num: int = 0,
                 obs_imu_his_num: int = 0,
                 config: Any = None,
                 num_envs: int = 4096):
        self.model_nv = model_nv
        self.default_pose = default_pose
        self.default_pose_mjx = default_pose_mjx
        self.isaac_to_mjx_idx = isaac_to_mjx_idx
        self.mjx_to_isaac_idx = mjx_to_isaac_idx
        self.torque_his_num = torque_his_num
        self.use_custom_pd = use_custom_pd
        self.kp_array = kp_array
        self.kd_array = kd_array
        self.torque_limits = torque_limits
        self.push_robots_fm = push_robots_fm
        self.obs_dof_his_num = obs_dof_his_num
        self.obs_imu_his_num = obs_imu_his_num
        self.config = config
        self.num_envs = num_envs
        
        # 获取地形类型
        self.terrain_type = 'plane'
        if config is not None:
            terrain_cfg = getattr(config, 'terrain', None)
            if terrain_cfg is not None:
                self.terrain_type = getattr(terrain_cfg, 'mesh_type', 'plane')

    def build(self, *, rng: jax.Array, randomize_dof_init: bool) -> dict[str, jax.Array]:
        """Build randomized reset qpos/qvel and initial torque history."""
        return build_reset_state(
            rng=rng, config=self.config, model_nv=self.model_nv,
            default_pose=self.default_pose,
            default_pose_mjx=self.default_pose_mjx,
            isaac_to_mjx_idx=self.isaac_to_mjx_idx,
            mjx_to_isaac_idx=self.mjx_to_isaac_idx,
            torque_his_num=self.torque_his_num,
            use_custom_pd=self.use_custom_pd,
            kp_array=self.kp_array, kd_array=self.kd_array,
            torque_limits=self.torque_limits,
            randomize_dof_init=randomize_dof_init,
            terrain_type=self.terrain_type,
            num_envs=self.num_envs,
        )

    def reset_info(self, *, info: dict[str, Any], done: jax.Array) -> dict[str, Any]:
        """Reset non-physics info fields when an episode terminates."""
        return reset_info_on_done(
            info=info, done=done,
            push_robots_fm=self.push_robots_fm,
        )


def build_reset_state(
    *,
    rng: jax.Array,
    config: Any,
    model_nv: int,
    default_pose: jax.Array,
    default_pose_mjx: jax.Array,
    isaac_to_mjx_idx: jax.Array,
    mjx_to_isaac_idx: jax.Array,
    torque_his_num: int,
    use_custom_pd: bool,
    kp_array: jax.Array,
    kd_array: jax.Array,
    torque_limits: jax.Array,
    randomize_dof_init: bool,
    terrain_type: str = 'plane',
    num_envs: int = 4096,
) -> dict[str, jax.Array]:
    """Build randomized reset qpos/qvel and initial torque history."""
    robot_cfg = getattr(config, "robot", None)
    init_jitter_height = getattr(robot_cfg, "init_jitter_height", 0.2)
    init_dof_noise = getattr(robot_cfg, "init_dof_noise", 0.1)
    init_dof_vel_noise = getattr(robot_cfg, "init_dof_vel_noise", 0.01)
    init_base_vel_noise = getattr(robot_cfg, "init_base_vel_noise", 0.5)

    rng, pos_rng, vel_rng, height_rng, terrain_rng = jax.random.split(rng, 5)

    init_pos = jp.array(config.init_state.pos)
    init_rot = jp.array(config.init_state.rot)
    random_height = jax.random.uniform(height_rng, (), minval=0.0, maxval=init_jitter_height)
    
    # ========== 多地形初始位置分配（使用 JAX 向量化操作） ==========
    # if terrain_type == "mixed":
    #     # 随机选择地形: 0=平地, 1=斜坡, 2=台阶, 3=凹凸
    #     terrain_choice = jax.random.randint(terrain_rng, (), 0, 4)
        
    #     # 生成4个候选位置
    #     rng_x = jax.random.split(terrain_rng, 8)
    #     rng_y = rng_x[4:8]
    #     rng_x = rng_x[:4]
        
    #     pos_x_0 = jax.random.uniform(rng_x[0], (), minval=-7.0, maxval=-3.0)  # 平地
    #     pos_y_0 = jax.random.uniform(rng_y[0], (), minval=-7.0, maxval=-3.0)
        
    #     pos_x_1 = jax.random.uniform(rng_x[1], (), minval=2.0, maxval=6.0)    # 斜坡
    #     pos_y_1 = jax.random.uniform(rng_y[1], (), minval=-7.0, maxval=-3.0)
        
    #     pos_x_2 = jax.random.uniform(rng_x[2], (), minval=-7.0, maxval=-3.0)  # 台阶
    #     pos_y_2 = jax.random.uniform(rng_y[2], (), minval=2.0, maxval=6.0)
        
    #     pos_x_3 = jax.random.uniform(rng_x[3], (), minval=2.0, maxval=6.0)    # 凹凸
    #     pos_y_3 = jax.random.uniform(rng_y[3], (), minval=2.0, maxval=6.0)
        
    #     # 使用 jnp.where 嵌套选择
    #     pos_x = jp.where(terrain_choice == 0, pos_x_0,
    #             jp.where(terrain_choice == 1, pos_x_1,
    #             jp.where(terrain_choice == 2, pos_x_2, pos_x_3)))
        
    #     pos_y = jp.where(terrain_choice == 0, pos_y_0,
    #             jp.where(terrain_choice == 1, pos_y_1,
    #             jp.where(terrain_choice == 2, pos_y_2, pos_y_3)))
        
    #     reset_init_pos = init_pos.at[0].set(init_pos[0] + pos_x)
    #     reset_init_pos = reset_init_pos.at[1].set(init_pos[1] + pos_y)
    #     reset_init_pos = reset_init_pos.at[2].set(init_pos[2] + random_height)
    if terrain_type == "mixed":
        # 随机选择地形: 0=平地, 1=斜坡, 2=台阶, 3=凹凸
        terrain_choice = jax.random.randint(terrain_rng, (), 0, 4)
        
        # 生成4个候选位置
        rng_x = jax.random.split(terrain_rng, 8)
        rng_y = rng_x[4:8]
        rng_x = rng_x[:4]
        
        # 平地 (左下): 中心在 (-5, -5)
        pos_x_0 = jax.random.uniform(rng_x[0], (), minval=-6.0, maxval=-4.0)
        pos_y_0 = jax.random.uniform(rng_y[0], (), minval=-6.0, maxval=-4.0)
        pos_z_0 = 0.45  # 平地不变
        
        # 斜坡 (右下): 中心在 (5, -5)，z轴下移0.45
        pos_x_1 = jax.random.uniform(rng_x[1], (), minval=4.0, maxval=6.0)
        pos_y_1 = jax.random.uniform(rng_y[1], (), minval=-6.0, maxval=-4.0)
        pos_z_1 = 0.0  # 下移0.45
        
        # 台阶 (左上): 中心在 (-5, 5)，z轴下移0.45
        pos_x_2 = jax.random.uniform(rng_x[2], (), minval=-6.0, maxval=-4.0)
        pos_y_2 = jax.random.uniform(rng_y[2], (), minval=4.0, maxval=6.0)
        pos_z_2 = 0.0  # 下移0.45
        
        # 凹凸 (右上): 中心在 (5, 5)
        pos_x_3 = jax.random.uniform(rng_x[3], (), minval=4.0, maxval=6.0)
        pos_y_3 = jax.random.uniform(rng_y[3], (), minval=4.0, maxval=6.0)
        pos_z_3 = 0.45  # 凹凸不变
        
        # 使用 jnp.where 嵌套选择
        pos_x = jp.where(terrain_choice == 0, pos_x_0,
                jp.where(terrain_choice == 1, pos_x_1,
                jp.where(terrain_choice == 2, pos_x_2, pos_x_3)))
        
        pos_y = jp.where(terrain_choice == 0, pos_y_0,
                jp.where(terrain_choice == 1, pos_y_1,
                jp.where(terrain_choice == 2, pos_y_2, pos_y_3)))
        
        pos_z = jp.where(terrain_choice == 0, pos_z_0,
                jp.where(terrain_choice == 1, pos_z_1,
                jp.where(terrain_choice == 2, pos_z_2, pos_z_3)))
        
        reset_init_pos = init_pos.at[0].set(init_pos[0] + pos_x)
        reset_init_pos = reset_init_pos.at[1].set(init_pos[1] + pos_y)
        reset_init_pos = reset_init_pos.at[2].set(init_pos[2] + random_height + pos_z)
            
    else:
        # 所有其他地形 (plane, slope, stairs, rough): 在原点附近 [-1, 1]
        pos_x = jax.random.uniform(terrain_rng, (), minval=-1.0, maxval=1.0)
        pos_y = jax.random.uniform(terrain_rng, (), minval=-1.0, maxval=1.0)
        reset_init_pos = init_pos.at[0].set(init_pos[0] + pos_x)
        reset_init_pos = reset_init_pos.at[1].set(init_pos[1] + pos_y)
        reset_init_pos = reset_init_pos.at[2].set(init_pos[2] + random_height)
    # ========================================================

    dof_noise = jax.random.uniform(pos_rng, (12,), minval=-init_dof_noise, maxval=init_dof_noise)
    joint_vel_noise = jax.random.uniform(
        vel_rng, (12,), minval=-init_dof_vel_noise, maxval=init_dof_vel_noise
    )

    if randomize_dof_init:
        reset_joint_pos = default_pose_mjx + dof_noise[isaac_to_mjx_idx]
    else:
        reset_joint_pos = default_pose_mjx
        joint_vel_noise = jp.zeros(12)

    reset_qpos = jp.concatenate([reset_init_pos, init_rot, reset_joint_pos])
    base_vel_noise = jax.random.uniform(
        vel_rng, (6,), minval=-init_base_vel_noise, maxval=init_base_vel_noise
    )
    reset_qvel = jp.zeros(model_nv)
    reset_qvel = reset_qvel.at[:6].set(base_vel_noise)
    reset_qvel = reset_qvel.at[6:18].set(joint_vel_noise)

    if use_custom_pd:
        reset_dof_pos = reset_joint_pos[mjx_to_isaac_idx]
        reset_dof_vel = jp.zeros(12)
        reset_torques = kp_array * (default_pose - reset_dof_pos) - kd_array * reset_dof_vel
        reset_torques = jp.clip(reset_torques, -torque_limits, torque_limits)
        reset_torque_his = jp.tile(reset_torques[:, None], (1, torque_his_num + 1))
    else:
        reset_torque_his = jp.zeros((12, torque_his_num + 1))

    return {
        "rng": rng,
        "qpos": reset_qpos,
        "qvel": reset_qvel,
        "torque_his": reset_torque_his,
    }


def reset_info_on_done(
    *,
    info: dict[str, Any],
    done: jax.Array,
    push_robots_fm: bool,
) -> dict[str, Any]:
    """Reset non-physics info fields when an episode terminates."""
    info["step"] = jp.where(done, 0, info["step"])
    info["unflip_time_out"] = jp.where(done, 0, info["unflip_time_out"])
    info["speed_unfollow_time_out"] = jp.where(done, 0, info["speed_unfollow_time_out"])
    info["cumulative_speed"] = jp.where(done, 0.0, info["cumulative_speed"])
    info["cumulative_cmd_speed"] = jp.where(done, 0.0, info["cumulative_cmd_speed"])
    info["low_height_time_out"] = jp.where(done, 0, info["low_height_time_out"])
    info["collision_time_out"] = jp.where(done, 0, info["collision_time_out"])
    info["ever_flipped"] = jp.where(done, False, info["ever_flipped"])
    info["feet_air_time"] = jp.where(done, jp.zeros(4), info["feet_air_time"])
    info["feet_contact_time"] = jp.where(done, jp.zeros(4), info["feet_contact_time"])
    info["last_action"] = jp.where(done, jp.zeros(12), info["last_action"])
    info["last_last_action"] = jp.where(done, jp.zeros(12), info["last_last_action"])
    info["last_dof_vel"] = jp.where(done, jp.zeros(12), info["last_dof_vel"])
    info["last_last_dof_vel"] = jp.where(done, jp.zeros(12), info["last_last_dof_vel"])
    info["last_base_lin_vel"] = jp.where(done, jp.zeros(3), info["last_base_lin_vel"])

    if push_robots_fm:
        info["push_FM_active"] = jp.where(done, False, info["push_FM_active"])
        info["push_FM_step_count"] = jp.where(done, 0, info["push_FM_step_count"])
        info["push_force"] = jp.where(done, jp.zeros(3), info["push_force"])
        info["push_torque"] = jp.where(done, jp.zeros(3), info["push_torque"])

    return info


__all__ = ["ResetSubComponent", "build_reset_state", "reset_info_on_done"]