"""XG quadruped robot specification."""

from __future__ import annotations

from pathlib import Path

from .base import BodyGroups, RobotSpec


XG_JOINT_NAMES = (
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
)


def build_xg_spec(project_root: Path) -> RobotSpec:
    """Build the current 12-DoF XG robot spec."""
    default_joint_angles = {
        "FBL_ABAD_JOINT": 0.0,
        "FAR_ABAD_JOINT": 0.0,
        "RBL_ABAD_JOINT": 0.0,
        "RAR_ABAD_JOINT": 0.0,
        "FBL_HIP_JOINT": 0.8,
        "FAR_HIP_JOINT": 0.8,
        "RBL_HIP_JOINT": 0.8,
        "RAR_HIP_JOINT": 0.8,
        "FBL_KNEE_JOINT": -1.5,
        "FAR_KNEE_JOINT": -1.5,
        "RBL_KNEE_JOINT": -1.5,
        "RAR_KNEE_JOINT": -1.5,
    }
    joint_lower = (-0.4887, -1.152, -2.723) * 4
    joint_upper = (0.4887, 2.967, -0.602) * 4
    identity = tuple(range(12))

    return RobotSpec(
        name="xg",
        asset_path=project_root / "../assets/bot_model/xg/xg.xml",
        num_dof=12,
        num_actions=12,
        joint_names=XG_JOINT_NAMES,
        default_joint_angles=default_joint_angles,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        torque_limits=(28.0,) * 12,
        body_groups=BodyGroups(
            feet=("FBL_FOOT_LINK", "FAR_FOOT_LINK", "RBL_FOOT_LINK", "RAR_FOOT_LINK"),
            penalized_contacts=(
                "torso",
                "FAR_HIP_LINK",
                "FBL_HIP_LINK",
                "RAR_HIP_LINK",
                "RBL_HIP_LINK",
                "FAR_KNEE_LINK",
                "FBL_KNEE_LINK",
                "RAR_KNEE_LINK",
                "RBL_KNEE_LINK",
                "FAR_FOOT_LINK",
                "FBL_FOOT_LINK",
                "RAR_FOOT_LINK",
                "RBL_FOOT_LINK",
            ),
            nonfoot_contacts=(
                "torso",
                "FAR_HIP_LINK",
                "FBL_HIP_LINK",
                "RAR_HIP_LINK",
                "RBL_HIP_LINK",
                "FAR_KNEE_LINK",
                "FBL_KNEE_LINK",
                "RAR_KNEE_LINK",
                "RBL_KNEE_LINK",
            ),
        ),
        mjx_to_policy_idx=identity,
        policy_to_mjx_idx=identity,
    )
