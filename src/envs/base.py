"""Base contracts for MJX environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax

from .state import State


class BaseEnv(ABC):
    """Minimal runtime contract shared by MJX environment backends."""

    config: Any

    @abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Create an initial environment state."""

    @abstractmethod
    def step(self, state: State, action: jax.Array, odom_data: jax.Array = None) -> State:
        """Advance one control step."""

    def render(self, state: State = None, states=None, sync_frame_time: bool = True):
        """Optional rendering hook for interactive backends."""
        raise NotImplementedError("This environment does not implement rendering.")


__all__ = ["BaseEnv"]
