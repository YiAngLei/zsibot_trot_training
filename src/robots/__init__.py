"""Robot specifications."""

from .base import BodyGroups, RobotSpec
from .xg import XG_JOINT_NAMES, build_xg_spec

__all__ = [
    "BodyGroups",
    "RobotSpec",
    "XG_JOINT_NAMES",
    "build_xg_spec",
]
