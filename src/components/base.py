"""Base contracts for environment component packages.

Only task-specific methods remain abstract here.  Mechanical methods
(physics stepping, contact tracking, history, reset, etc.) are provided
by ``MechanicalBase`` which task components should inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseComponent(ABC):
    """Task component contract used by generic environment backends.

    Subclasses implement task-specific logic for commands, observations,
    rewards, and terminations.  Mechanical orchestration (physics stepping,
    contact tracking, history buffers, etc.) is provided by
    :class:`MechanicalBase`.
    """

    name: str
    reward_terms: Mapping[str, Any] = {}
    termination_terms: Mapping[str, Any] = {}
    required_rewards: tuple[str, ...] = ()
    required_terminations: tuple[str, ...] = ()
    external_reward_names: tuple[str, ...] = ("termination",)

    # ------------------------------------------------------------------
    # Task-specific abstract methods — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def sample_command(self, **kwargs: Any) -> Any:
        """Sample a command for the current task."""

    @abstractmethod
    def build_observation(self, **kwargs: Any) -> Mapping[str, Any]:
        """Build actor and privileged observations."""

    @abstractmethod
    def build_reward_context(self, **kwargs: Any) -> Any:
        """Build the task-specific reward context."""

    @abstractmethod
    def compute_rewards(self, **kwargs: Any) -> dict[str, Any]:
        """Compute active task rewards."""

    def compute_foot_swing_height_reward(self, **kwargs: Any) -> Any:
        """Optional stateful reward hook used by velocity-style components."""
        raise NotImplementedError(
            f"Component '{self.name}' does not implement foot swing height rewards."
        )

    def compute_stateful_rewards(self, *, reward_scales, info, data, **kwargs: Any) -> tuple[dict, dict]:
        """Compute rewards that require cross-step state (e.g. peak-height tracking).

        Returns ``(extra_rewards_dict, info_updates_dict)`` where
        ``extra_rewards_dict`` maps reward name to unscaled reward value and
        ``info_updates_dict`` maps info key to updated value.

        Default: no stateful rewards.  Override in task components that need it.
        """
        return {}, {}

    @abstractmethod
    def compute_terminations(self, **kwargs: Any) -> Mapping[str, Any]:
        """Compute task termination flags."""

    @abstractmethod
    def write_termination_info(self, **kwargs: Any) -> Mapping[str, Any]:
        """Write termination flags back into environment info."""

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate that declared required terms exist in this component."""
        missing_rewards = [
            name for name in self.required_rewards if name not in self.reward_terms
        ]
        missing_terminations = [
            name for name in self.required_terminations if name not in self.termination_terms
        ]
        if missing_rewards or missing_terminations:
            details = []
            if missing_rewards:
                details.append(f"missing rewards: {', '.join(missing_rewards)}")
            if missing_terminations:
                details.append(f"missing terminations: {', '.join(missing_terminations)}")
            raise ValueError(f"Invalid component '{self.name}' ({'; '.join(details)}).")

    def validate_reward_scales(self, reward_scales: Mapping[str, float]) -> None:
        """Validate active configured reward names against this component."""
        allowed = set(self.reward_terms) | set(self.external_reward_names)
        unknown = sorted(name for name in reward_scales if name not in allowed)
        if unknown:
            raise ValueError(
                f"Component '{self.name}' does not define active reward(s): "
                + ", ".join(unknown)
            )


__all__ = ["BaseComponent"]
