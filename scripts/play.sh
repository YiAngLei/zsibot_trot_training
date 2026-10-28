# python src/play.py --exp xg_velocity --checkpoint assets/nn_model/mlp/iteration_1800.pkl
# python src/play.py --exp xg_velocity --checkpoint checkpoints/xg_velocity/checkpoints_20260609_180443/iteration_5000.pkl
# python src/play.py --exp xg_velocity --checkpoint assets/nn_model/mlp/iteration_1800.pkl
# JAX_PLATFORMS=cpu python src/ros_play.py --checkpoint checkpoints/xg_velocity/checkpoints_20260821_155536/iteration_1600.pkl
#!/bin/bash
source /home/ubuntu20/miniconda3/etc/profile.d/conda.sh
cd project_4_DR_trot_training/code/
conda activate mjx_ros
JAX_PLATFORMS=cpu python src/ros_play.py --checkpoint assets/nn_model/mlp/iteration_2100.pkl
# JAX_PLATFORMS=cpu python src/ros_play.py --checkpoint checkpoints/xg_velocity/checkpoints_20260915_141711/iteration_4200.pkl
