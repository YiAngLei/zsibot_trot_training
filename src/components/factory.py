"""Component factory."""

from __future__ import annotations

from typing import Any

from .base import BaseComponent


def create_component(name: str, **kwargs: Any) -> BaseComponent:
    """Factory function: create a component instance by name.

    Args:
        name: Registered component name (e.g. ``"velocity"``).
        **kwargs: Forwarded to the component constructor.

    Returns:
        A validated ``BaseComponent`` instance.

    Raises:
        KeyError: If *name* is not a known component.
    """
    # Lazy import so that all task modules are registered before lookup.
    from .tasks import _TASK_REGISTRY  # noqa: F401

    if name not in _TASK_REGISTRY:
        raise KeyError(
            f"Unknown component '{name}'. Available: {sorted(_TASK_REGISTRY)}"
        )
    component = _TASK_REGISTRY[name](**kwargs)
    component.validate()
    return component


def list_components() -> tuple[str, ...]:
    """Return a sorted tuple of all registered component names."""
    from .tasks import _TASK_REGISTRY  # noqa: F401

    return tuple(sorted(_TASK_REGISTRY))


__all__ = ["create_component", "list_components"]
