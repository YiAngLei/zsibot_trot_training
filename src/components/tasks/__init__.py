"""Task components registry."""

from .velocity import VelocityComponent

_TASK_REGISTRY: dict[str, type] = {
    "velocity": VelocityComponent,
}

__all__ = ["VelocityComponent", "_TASK_REGISTRY"]
