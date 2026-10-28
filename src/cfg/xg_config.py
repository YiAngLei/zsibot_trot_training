"""
Configuration for XG quadruped robot environment in MJX.

Velocity Go1 Version
========================================
- Observation: base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command
- Action: joint position targets with scale 0.5
- Rewards
"""

import math
from ml_collections import config_dict


def get_xg_config():
    config = config_dict.ConfigDict()
    
    # ========== Environment settings ==========
    config.env = config_dict.ConfigDict()
    config.env.num_envs = 4096
    # velocity obs: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + 
    #               joint_pos(12) + joint_vel(12) + actions(12) + command(3) = 48
    config.env.num_observations = 48
    config.env.num_actions = 12
    config.env.obs_history_length = 1

    # episode_length_s=20.0, decimation=4, dt=0.005 -> 20/(0.005*4) = 1000 steps
    config.env.episode_length = 1000
    
    # ========== Commands configuration ==========
    config.commands = config_dict.ConfigDict()
    config.commands.num_commands = 3  # lin_vel_x, lin_vel_y, ang_vel_z
    # resampling_time_range=(3.0, 8.0), use the midpoint 5.5s
    config.commands.resampling_time = 5.5

    # Curriculum is not used (default curriculum in env_cfgs.py is unchanged)
    config.commands.curriculum = False
    config.commands.max_curriculum = 3.0
    config.commands.curriculum_threshold = 0.8
    config.commands.curriculum_step_size = 0.5
    
    config.commands.ranges = config_dict.ConfigDict()
    config.commands.ranges.lin_vel_x = (-1.0, 1.0)
    config.commands.ranges.lin_vel_y = (-1.0, 1.0)
    config.commands.ranges.ang_vel_yaw = (-0.5, 0.5)
    config.commands.ranges.heading = (-math.pi, math.pi)
    config.commands.heading_command = True
    config.commands.heading_control_stiffness = 0.5
    
    config.commands.zero_cmd_probability = 0.1
    config.commands.rel_heading_envs = 0.3
    
    # ========== Initial state ==========
    config.init_state = config_dict.ConfigDict()
    config.init_state.pos = (0.0, 0.0, 0.35)  # Go1 standing height ~ 0.35 m
    config.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    # Default Go1 joint angles
    config.init_state.default_joint_angles = config_dict.ConfigDict()
    config.init_state.default_joint_angles.FBL_ABAD_JOINT = 0.0   # FL_hip
    config.init_state.default_joint_angles.FAR_ABAD_JOINT = 0.0   # FR_hip
    config.init_state.default_joint_angles.RBL_ABAD_JOINT = 0.0   # RL_hip
    config.init_state.default_joint_angles.RAR_ABAD_JOINT = 0.0   # RR_hip
    config.init_state.default_joint_angles.FBL_HIP_JOINT = 0.8    # FL_thigh
    config.init_state.default_joint_angles.FAR_HIP_JOINT = 0.8    # FR_thigh
    config.init_state.default_joint_angles.RBL_HIP_JOINT = 0.8    # RL_thigh
    config.init_state.default_joint_angles.RAR_HIP_JOINT = 0.8    # RR_thigh
    config.init_state.default_joint_angles.FBL_KNEE_JOINT = -1.5  # FL_calf
    config.init_state.default_joint_angles.FAR_KNEE_JOINT = -1.5  # FR_calf
    config.init_state.default_joint_angles.RBL_KNEE_JOINT = -1.5  # RL_calf
    config.init_state.default_joint_angles.RAR_KNEE_JOINT = -1.5  # RR_calf
    
    # ========== Control configuration ==========
    config.control = config_dict.ConfigDict()
    config.control.control_type = 'P'
    config.control.stiffness = 20.0   # Default Go1 PD parameter
    config.control.damping = 0.5
    config.control.action_scale = 0.5
    config.control.decimation = 4
    config.control.use_custom_pd = True
    config.control.action_clip = 1.0  # action clipping upper bound

    # ========== Robot geometry (extracted from env hardcode) ==========
    config.robot = config_dict.ConfigDict()
    # Go1 joint limits (in 4x3 order: FR, FL, RR, RL; each leg ABAD/HIP/KNEE)
    config.robot.joint_lower = (
        -0.4887, -1.152, -2.723,   # FR (FAR)
        -0.4887, -1.152, -2.723,   # FL (FBL)
        -0.4887, -1.152, -2.723,   # RR (RAR)
        -0.4887, -1.152, -2.723,   # RL (RBL)
    )
    config.robot.joint_upper = (
        0.4887, 2.967, -0.602,    # FR
        0.4887, 2.967, -0.602,    # FL
        0.4887, 2.967, -0.602,    # RR
        0.4887, 2.967, -0.602,    # RL
    )
    config.robot.torque_limits = 28.0  # Nm
    config.robot.dof_vel_limits = 28.0  # rad/s
    config.robot.num_joints = 12

    # Body/site names (in FR, FL, RR, RL order)
    config.robot.foot_body_names = (
        'FBL_FOOT_LINK', 'FAR_FOOT_LINK', 'RBL_FOOT_LINK', 'RAR_FOOT_LINK',
    )
    config.robot.foot_site_names = (
        'fl_foot_site', 'fr_foot_site', 'rl_foot_site', 'rr_foot_site',
    )
    config.robot.nonfoot_body_names = (
        'torso', 'FAR_HIP_LINK', 'FBL_HIP_LINK', 'RAR_HIP_LINK', 'RBL_HIP_LINK',
        'FAR_KNEE_LINK', 'FBL_KNEE_LINK', 'RAR_KNEE_LINK', 'RBL_KNEE_LINK',
    )

    # Initial-state noise scales
    config.robot.init_jitter_height = 0.2
    config.robot.init_dof_noise = 0.1
    config.robot.init_dof_vel_noise = 0.01
    config.robot.init_base_vel_noise = 0.5

    # Observation sizes (for assertion / logging)
    config.robot.num_actor_obs = 48
    config.robot.num_privileged_obs = 259  # 72 原有维度 + 11×17 地形高度扫描(187)
    config.robot.odom_obs_size = 29
    
    # ========== Simulation configuration ==========
    config.sim = config_dict.ConfigDict()
    config.sim.dt = 0.005
    config.sim.substeps = 1
    config.use_soft_contact = False
    config.sim.nconmax = 4 * 8192
    config.sim.njmax = 300 
    config.sim.impl = "jax"
    
    # ========== Domain randomization ==========
    config.domain_rand = config_dict.ConfigDict()
    
    config.domain_rand.randomize_friction = True
    config.domain_rand.friction_range = (0.3, 1.2)
    
    config.domain_rand.randomize_com_offset = True
    config.domain_rand.com_offset_range = ((-0.025, 0.025), (-0.025, 0.025), (-0.03, 0.03))
    
    config.domain_rand.randomize_encoder_bias = True
    config.domain_rand.encoder_bias_range = (-0.015, 0.015)
    
    config.domain_rand.push_robots = True
    config.domain_rand.push_interval_s = 2.0  # ~1-3 s average interval
    config.domain_rand.push_duration_s = 0.0  # instantaneous push
    config.domain_rand.max_push_vel_xy = 0.5  # velocity_range x,y = (-0.5, 0.5)
    config.domain_rand.max_push_vel_z = 0.4   # velocity_range z = (-0.4, 0.4)
    config.domain_rand.max_push_ang_vel = 0.52  # roll/pitch = (-0.52, 0.52)
    config.domain_rand.max_push_ang_vel_yaw = 0.78  # yaw = (-0.78, 0.78)

    # Other randomization terms are temporarily disabled
    config.domain_rand.randomize_base_mass = False
    config.domain_rand.added_mass_range = (-1.0, 1.0)
    config.domain_rand.randomize_mass = False
    config.domain_rand.mass_factor_range = (0.9, 1.1)
    config.domain_rand.randomize_motor_strength = False
    config.domain_rand.motor_strength_range = (0.9, 1.1)
    config.domain_rand.randomize_Kp_factor = False
    config.domain_rand.Kp_factor_range = (0.9, 1.1)
    config.domain_rand.randomize_Kd_factor = False
    config.domain_rand.Kd_factor_range = (0.9, 1.1)
    config.domain_rand.randomize_dof_init = True
    config.domain_rand.randomize_inertia = False
    config.domain_rand.inertia_factor_range = (0.9, 1.1)
    config.domain_rand.randomize_dof_damping = False
    config.domain_rand.dof_damping_range = (0.0, 0.2)
    config.domain_rand.randomize_action_latency = False
    config.domain_rand.action_latency_range = (0.0, 0.002)
    config.domain_rand.randomize_obs_latency = False
    config.domain_rand.obs_dof_his_num = 1
    config.domain_rand.obs_imu_his_num = 1
    config.domain_rand.obs_latency_steps = 1
    config.domain_rand.randomize_torques_latency = False
    config.domain_rand.torque_his_num = 3
    config.domain_rand.randomize_restitution = False
    config.domain_rand.restitution_range = (0.0, 0.3)
    
    # Disable FM push force
    config.domain_rand.push_robots_FM = False
    config.domain_rand.push_FM_interval_s = 15.0
    config.domain_rand.push_FM_duration_s = 1.0
    config.domain_rand.max_push_force_FM = 50.0
    config.domain_rand.max_push_torque_FM = 5.0
    config.domain_rand.kick_vel = 1.5
    
    # ========== Terrain configuration ==========
    config.terrain = config_dict.ConfigDict()
    config.terrain.mesh_type = 'stairs'  # 'plane' | 'slope' | 'stairs' | 'rough' | 'mixed'
    config.terrain.slope_angle = 10.0   # 度
    config.terrain.stair_height = 0.05  # 米
    config.terrain.rough_range = (-0.03, 0.03)  # 米
    config.terrain.measure_heights = True
    config.terrain.horizontal_scale = 0.1
    config.terrain.vertical_scale = 0.005
    config.terrain.num_terrain_rows = 4
    config.terrain.num_terrain_cols = 4
    
    # ========== Rewards configuration ==========
    config.rewards = config_dict.ConfigDict()
    config.rewards.tracking_sigma_lin = 0.5
    config.rewards.tracking_sigma_ang = math.sqrt(0.5)
    config.rewards.upright_sigma = math.sqrt(0.2)
    # Gait parameters
    config.rewards.feet_air_time_target_min = 0.05
    config.rewards.feet_air_time_target_max = 0.5
    config.rewards.foot_clearance_target = 0.15
    config.rewards.command_threshold = 0.05
    
    # ========== pose (variable_posture) ==========
    config.rewards.pose_walking_threshold = 0.05
    config.rewards.pose_running_threshold = 1.5
    config.rewards.pose_std_standing_hip = 0.05   # r".*(hip|thigh)_joint.*": 0.05
    config.rewards.pose_std_standing_calf = 0.1   # r".*calf_joint.*": 0.1
    config.rewards.pose_std_walking_hip = 0.3     # r".*(hip|thigh)_joint.*": 0.3
    config.rewards.pose_std_walking_calf = 0.6    # r".*calf_joint.*": 0.6
    config.rewards.pose_std_running_hip = 0.3     # r".*(hip|thigh)_joint.*": 0.3
    config.rewards.pose_std_running_calf = 0.6    # r".*calf_joint.*": 0.6
    
    # ========== Reward Scales ==========
    config.rewards.scales = config_dict.ConfigDict()
    
    # ========== Velocity Track ==========
    config.rewards.scales.tracking_lin_vel = 2.0
    config.rewards.scales.tracking_ang_vel = 2.0
    config.rewards.scales.lateral_vel = -0.2
    config.rewards.scales.yaw_rate_error = -0.2
    
    # ========== Pose Reward ==========
    config.rewards.scales.upright = 1.0
    config.rewards.scales.pose = 1.0
    
    # ========== Joint limit configuration ==========
    # soft_dof_pos_limit: ratio of the joint range used as a soft limit (0.9 = 90% of the range)
    config.rewards.soft_dof_pos_limit = 0.9
    config.rewards.scales.dof_pos_limits = -1.0

    # ========== Action smoothness ==========
    config.rewards.scales.action_rate = -0.1

    # ========== Gait reward / penalty ==========
    config.rewards.scales.feet_air_time = 0.0
    config.rewards.scales.foot_clearance = -2.0
    config.rewards.scales.foot_swing_height = -0.25
    config.rewards.scales.slip = -0.1
    config.rewards.scales.soft_landing = -1e-5

    # ========== Other rewards in the Go1 config with weight=0 ==========
    config.rewards.scales.body_ang_vel = 0.0
    config.rewards.scales.angular_momentum = 0.0

    # ========== Disabled task-specific rewards ==========
    config.rewards.scales.orientation = 0.0
    config.rewards.scales.base_height = 0.0
    config.rewards.scales.vel_mismatch_exp = 0.0
    config.rewards.scales.base_lin_acc = 0.0
    config.rewards.scales.dof_pos_abad = 0.0
    config.rewards.scales.torques = 0.0
    config.rewards.scales.power = 0.0
    config.rewards.scales.dof_acc = 0.0
    config.rewards.scales.power_distribution = 0.0
    config.rewards.scales.smoothness = 0.0
    config.rewards.scales.dof_vel_limits = 0.0
    config.rewards.scales.torque_limits = 0.0
    config.rewards.scales.collision = 0.0
    config.rewards.scales.stand_still = 0.0
    config.rewards.scales.termination = 0.0
    config.rewards.scales.stumble = 0.0
    config.rewards.scales.feet_contact_forces = 0.0
    
    config.rewards.only_positive_rewards = False
    
    # ========== Observation scales ==========
    config.obs_scales = config_dict.ConfigDict()
    config.obs_scales.lin_vel = 1.0
    config.obs_scales.ang_vel = 1.0
    config.obs_scales.dof_pos = 1.0
    config.obs_scales.dof_vel = 1.0
    config.obs_scales.projected_gravity = 1.0
    
    # ========== Noise configuration ==========
    config.noise = config_dict.ConfigDict()
    config.noise.add_noise = True
    config.noise.noise_level = 1.0
    config.noise.scales = config_dict.ConfigDict()
    config.noise.scales.dof_pos = 0.01
    config.noise.scales.dof_vel = 1.5
    config.noise.scales.lin_vel = 0.5
    config.noise.scales.ang_vel = 0.2
    config.noise.scales.gravity = 0.05
    
    # ========== Perturbation configuration ==========
    config.pert_config = config_dict.ConfigDict()
    config.pert_config.enable = False
    config.pert_config.velocity_kick = (3.0, 6.0)
    config.pert_config.kick_wait_times = (5.0, 15.0)
    
    # ========== Termination configuration ==========
    config.termination = config_dict.ConfigDict()
    config.termination.fell_over_limit_angle = math.radians(70.0)
    config.termination.illegal_contact_force_threshold = 10.0
    
    return config


def get_xg_ppo_config():
    config = config_dict.ConfigDict()
    
    # ========== Training parameters ==========
    config.max_iterations = 10000 
    config.num_timesteps = 200_000_000
    config.num_envs = 4096
    
    config.num_evals = 100
    
    # ========== Reward Scaling configuration ==========
    config.use_isaac_reward_scaling = False
    config.reward_scaling_gamma = 0.99
    config.reward_scaling = 1.0  # no scaling

    config.episode_length = 1000
    config.normalize_observations = False
    config.action_repeat = 1

    # ========== PPO parameters ==========
    config.unroll_length = 24
    config.num_minibatches = 4
    config.num_updates_per_batch = 5
    config.discounting = 0.99
    config.learning_rate = 1e-3
    config.entropy_cost = 0.01
    config.max_grad_norm = 1.0
    config.gae_lambda = 0.95
    config.clip_param = 0.2
    config.desired_kl = 0.01
    config.value_loss_coef = 1.0
    config.use_clipped_value_loss = True
    config.lr_schedule = "adaptive"
    
    # ========== Network architecture ==========
    config.network = config_dict.ConfigDict()
    config.network.policy_hidden_layer_sizes = (512, 256, 128)
    config.network.value_hidden_layer_sizes = (512, 256, 128)
    config.network.activation = 'elu'
    config.network.init_noise_std = 1.0
    
    # ========== Checkpointing ==========
    config.save_interval = 50
    config.checkpoint_save_interval_steps = 10_000_000
    
    return config
