"""Velocity observation schema and builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jp
from brax import math


@dataclass(frozen=True)
class ObservationField:
    name: str
    size: int
    start: int

    @property
    def stop(self) -> int:
        return self.start + self.size


# ========== 地形高度扫描网格配置 ==========
# 以基座为中心的规则网格：纵向 x（机器人前进方向）17 点覆盖 1.6 m，
# 横向 y 11 点覆盖 1.0 m，分辨率 0.1 m，共 187 点
HEIGHT_SCAN_NUM_X = 17
HEIGHT_SCAN_NUM_Y = 11
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_NUM_POINTS = HEIGHT_SCAN_NUM_X * HEIGHT_SCAN_NUM_Y  # 187


VELOCITY_ACTOR_SCHEMA = (
    ObservationField("base_lin_vel", 3, 0),
    ObservationField("base_ang_vel", 3, 3),
    ObservationField("projected_gravity", 3, 6),
    ObservationField("joint_pos_rel", 12, 9),
    ObservationField("joint_vel", 12, 21),
    ObservationField("last_action", 12, 33),
    ObservationField("command", 3, 45),
)

VELOCITY_PRIVILEGED_SCHEMA = VELOCITY_ACTOR_SCHEMA + (
    ObservationField("foot_height", 4, 48),
    ObservationField("foot_air_time", 4, 52),
    ObservationField("foot_contact", 4, 56),
    ObservationField("foot_contact_forces", 12, 60),
    ObservationField("height_scan", HEIGHT_SCAN_NUM_POINTS, 72),
)


def build_velocity_observation(
    *,
    data: Any,
    info: Mapping[str, Any],
    action,
    default_pose,
    mjx_to_policy_idx,
    feet_body_ids,
    randomize_obs_latency: bool,
    obs_imu_his_num: int,
    obs_dof_his_num: int,
    add_noise: bool,
    noise_config: Any,
) -> dict[str, Any]:
    """Build current velocity actor and privileged observations."""
    base_quat = data.qpos[3:7]
    mjx_joint_pos = data.qpos[7:19]
    mjx_joint_vel = data.qvel[6:18]
    base_lin_vel_world = data.qvel[:3]
    base_ang_vel_world = data.qvel[3:6]

    inv_rot = math.quat_inv(base_quat)
    base_lin_vel = math.rotate(base_lin_vel_world, inv_rot)
    base_ang_vel = math.rotate(base_ang_vel_world, inv_rot)
    projected_gravity = math.rotate(jp.array([0.0, 0.0, -1.0]), inv_rot)

    joint_pos = mjx_joint_pos[mjx_to_policy_idx]
    joint_vel = mjx_joint_vel[mjx_to_policy_idx]

    if randomize_obs_latency:
        his_ang_vel = info["ang_lin_history"][:, obs_imu_his_num]
        his_dof_pos = info["dof_pos_history"][:, obs_dof_his_num]
        his_dof_vel = info["dof_vel_history"][:, obs_dof_his_num]
    else:
        his_ang_vel = base_ang_vel
        his_dof_pos = joint_pos
        his_dof_vel = joint_vel

    obs_lin_vel = info.get("odom_data", base_lin_vel)

    obs = jp.concatenate(
        [
            obs_lin_vel,
            his_ang_vel,
            projected_gravity,
            his_dof_pos - default_pose,
            his_dof_vel,
            action,
            info["command"],
        ]
    )

    if add_noise:
        rng = info.get("rng", jax.random.PRNGKey(0))
        noise_scales = noise_config.scales
        noise_lin_vel = jax.random.uniform(
            rng, (3,), minval=-noise_scales.lin_vel, maxval=noise_scales.lin_vel
        )
        rng, _ = jax.random.split(rng)
        noise_ang_vel = jax.random.uniform(
            rng, (3,), minval=-noise_scales.ang_vel, maxval=noise_scales.ang_vel
        )
        rng, _ = jax.random.split(rng)
        noise_gravity = jax.random.uniform(
            rng, (3,), minval=-noise_scales.gravity, maxval=noise_scales.gravity
        )
        rng, _ = jax.random.split(rng)
        noise_dof_pos = jax.random.uniform(
            rng, (12,), minval=-noise_scales.dof_pos, maxval=noise_scales.dof_pos
        )
        rng, _ = jax.random.split(rng)
        noise_dof_vel = jax.random.uniform(
            rng, (12,), minval=-noise_scales.dof_vel, maxval=noise_scales.dof_vel
        )

        noise = jp.concatenate(
            [
                noise_lin_vel,
                noise_ang_vel,
                noise_gravity,
                noise_dof_pos,
                noise_dof_vel,
                jp.zeros(12),
                jp.zeros(3),
            ]
        )
        obs = obs + noise

    obs = jp.clip(obs, -100.0, 100.0)

    feet_forces = data.cfrc_ext[feet_body_ids, 3:6]
    contact_force_z = feet_forces[:, 2]
    contacts = (contact_force_z > 1.0).astype(jp.float32)
    foot_heights = data.xpos[feet_body_ids, 2]
    feet_air_time = info.get("feet_air_time", jp.zeros(4))
    contact_forces = feet_forces.flatten()

    # ========== 地形高度扫描：以基座为中心的 11×17 规则网格 ==========
    # 纵向 x（前进方向）17 点覆盖 1.6 m，横向 y 11 点覆盖 1.0 m，分辨率 0.1 m
    # 展平顺序为行主序：y 为行、x 为列，reshape(HEIGHT_SCAN_NUM_Y, HEIGHT_SCAN_NUM_X) 可还原网格
    x_points = jp.linspace(
        -(HEIGHT_SCAN_NUM_X - 1) / 2 * HEIGHT_SCAN_RESOLUTION,
        (HEIGHT_SCAN_NUM_X - 1) / 2 * HEIGHT_SCAN_RESOLUTION,
        HEIGHT_SCAN_NUM_X,
    )  # -0.8 ~ 0.8
    y_points = jp.linspace(
        -(HEIGHT_SCAN_NUM_Y - 1) / 2 * HEIGHT_SCAN_RESOLUTION,
        (HEIGHT_SCAN_NUM_Y - 1) / 2 * HEIGHT_SCAN_RESOLUTION,
        HEIGHT_SCAN_NUM_Y,
    )  # -0.5 ~ 0.5
    grid_x, grid_y = jp.meshgrid(x_points, y_points, indexing="xy")  # (11, 17)
    sample_offsets = jp.stack(
        [grid_x.ravel(), grid_y.ravel(), jp.zeros(HEIGHT_SCAN_NUM_POINTS)], axis=-1
    )  # (187, 3)

    # 获取基座位置（3维）
    base_pos = data.xpos[0]
    base_quat = data.qpos[3:7]

    # 将偏移量旋转到世界坐标系（使用四元数），vmap 向量化避免逐点循环
    rotated_offsets = jax.vmap(lambda offset: math.rotate(offset, base_quat))(sample_offsets)

    # 计算采样点的世界坐标（只需要 x, y 用于查询地形）
    sample_positions = base_pos[None, :2] + rotated_offsets[:, :2]

    # 查询地形高度（vmap 向量化双线性插值查询）
    query_fn = info.get("terrain_query")
    if query_fn is not None:
        height_scan = jax.vmap(query_fn)(sample_positions[:, 0], sample_positions[:, 1])
    else:
        # 无地形时返回0
        height_scan = jp.zeros(HEIGHT_SCAN_NUM_POINTS)


    privileged_obs = jp.concatenate(
        [
            obs_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos - default_pose,
            joint_vel,
            action,
            info["command"],
            foot_heights,
            feet_air_time,
            contacts,
            contact_forces,
            height_scan,
        ]
    )
    privileged_obs = jp.clip(privileged_obs, -100.0, 100.0)

    return {"state": obs, "privileged": privileged_obs}
