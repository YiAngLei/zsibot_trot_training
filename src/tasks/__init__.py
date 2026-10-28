"""Task specifications."""

from .base import ObservationSpec, TaskSpec
from .velocity import build_velocity_task

__all__ = [
    "ObservationSpec",
    "TaskSpec",
    "build_velocity_task",
]
