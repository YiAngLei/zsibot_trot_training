"""Compatibility layer for EnvInfo <-> dict[str, Array] conversion.

Transitional utility: once all callers use ``EnvInfo`` attribute access
directly, this module can be deleted.
"""

from __future__ import annotations

from .info import ContactState, EnvInfo, HistoryBuffers, PerturbationState


def info_to_dict(info: EnvInfo) -> dict:
    """Flatten an ``EnvInfo`` into a plain dict of arrays.

    JAX-pure (no Python control flow, only structural access).
    The returned dict uses the same flat key names that sub-components
    currently expect (e.g. ``"feet_air_time"`` rather than
    ``info.contact.feet_air_time``).
    """
    return {
        # Counters
        "step": info.step,
        "common_step_counter": info.common_step_counter,
        # Command
        "command": info.command,
        "heading_target": info.heading_target,
        # DR factors
        "motor_strength": info.motor_strength,
        "kp_factor": info.kp_factor,
        "kd_factor": info.kd_factor,
        "dof_damping": info.dof_damping,
        "friction_coeff": info.friction_coeff,
        "inertia_factors": info.inertia_factors,
        "mass_factors": info.mass_factors,
        "com_offset": info.com_offset,
        "restitution_coeff": info.restitution_coeff,
        "feet_friction": info.feet_friction,
        "action_latency": info.action_latency,
        # Contact
        "last_contact": info.contact.last_contact,
        "feet_air_time": info.contact.feet_air_time,
        "feet_contact_time": info.contact.feet_contact_time,
        "peak_heights": info.contact.peak_heights,
        # History
        "dof_pos_history": info.history.dof_pos_history,
        "dof_vel_history": info.history.dof_vel_history,
        "rpy_history": info.history.rpy_history,
        "ang_lin_history": info.history.ang_lin_history,
        "torque_his": info.history.torque_his,
        # Perturbation
        "push_FM_active": info.perturbation.push_FM_active,
        "push_FM_step_count": info.perturbation.push_FM_step_count,
        "push_force": info.perturbation.push_force,
        "push_torque": info.perturbation.push_torque,
        # Control
        "last_action": info.last_action,
        "last_last_action": info.last_last_action,
        "last_dof_vel": info.last_dof_vel,
        "last_last_dof_vel": info.last_last_dof_vel,
        "last_base_lin_vel": info.last_base_lin_vel,
        # RNG
        "rng": info.rng,
        # Termination
        "done_fell_over": info.done_fell_over,
        "done_illegal_contact": info.done_illegal_contact,
        "done_nan": info.done_nan,
        "time_out": info.time_out,
        "done_unflip": info.done_unflip,
        "unflip_time_out": info.unflip_time_out,
        "speed_unfollow_time_out": info.speed_unfollow_time_out,
        "low_height_time_out": info.low_height_time_out,
        "collision_time_out": info.collision_time_out,
        "ever_flipped": info.ever_flipped,
        # Odometry
        "odom_data": info.odom_data,
        "odom_ground_truth": info.odom_ground_truth,
        # Reward
        "rewards": info.rewards,
        # Cumulative
        "cumulative_speed": info.cumulative_speed,
        "cumulative_cmd_speed": info.cumulative_cmd_speed,
    }


def dict_to_info(d: dict) -> EnvInfo:
    """Reconstruct an ``EnvInfo`` from a flat dict.

    Inverse of ``info_to_dict``: nested sub-structs are assembled from
    their flat key names.
    """
    return EnvInfo(
        # Counters
        step=d["step"],
        common_step_counter=d["common_step_counter"],
        # Command
        command=d["command"],
        heading_target=d["heading_target"],
        # DR factors
        motor_strength=d["motor_strength"],
        kp_factor=d["kp_factor"],
        kd_factor=d["kd_factor"],
        dof_damping=d["dof_damping"],
        friction_coeff=d["friction_coeff"],
        inertia_factors=d["inertia_factors"],
        mass_factors=d["mass_factors"],
        com_offset=d["com_offset"],
        restitution_coeff=d["restitution_coeff"],
        feet_friction=d["feet_friction"],
        action_latency=d["action_latency"],
        # Contact
        contact=ContactState(
            last_contact=d["last_contact"],
            feet_air_time=d["feet_air_time"],
            feet_contact_time=d["feet_contact_time"],
            peak_heights=d["peak_heights"],
        ),
        # History
        history=HistoryBuffers(
            dof_pos_history=d["dof_pos_history"],
            dof_vel_history=d["dof_vel_history"],
            rpy_history=d["rpy_history"],
            ang_lin_history=d["ang_lin_history"],
            torque_his=d["torque_his"],
        ),
        # Perturbation
        perturbation=PerturbationState(
            push_FM_active=d["push_FM_active"],
            push_FM_step_count=d["push_FM_step_count"],
            push_force=d["push_force"],
            push_torque=d["push_torque"],
        ),
        # Control
        last_action=d["last_action"],
        last_last_action=d["last_last_action"],
        last_dof_vel=d["last_dof_vel"],
        last_last_dof_vel=d["last_last_dof_vel"],
        last_base_lin_vel=d["last_base_lin_vel"],
        # RNG
        rng=d["rng"],
        # Termination
        done_fell_over=d["done_fell_over"],
        done_illegal_contact=d["done_illegal_contact"],
        done_nan=d["done_nan"],
        time_out=d["time_out"],
        done_unflip=d["done_unflip"],
        unflip_time_out=d["unflip_time_out"],
        speed_unfollow_time_out=d["speed_unfollow_time_out"],
        low_height_time_out=d["low_height_time_out"],
        collision_time_out=d["collision_time_out"],
        ever_flipped=d["ever_flipped"],
        # Odometry
        odom_data=d["odom_data"],
        odom_ground_truth=d["odom_ground_truth"],
        # Reward
        rewards=d["rewards"],
        # Cumulative
        cumulative_speed=d["cumulative_speed"],
        cumulative_cmd_speed=d["cumulative_cmd_speed"],
    )


__all__ = ["dict_to_info", "info_to_dict"]
