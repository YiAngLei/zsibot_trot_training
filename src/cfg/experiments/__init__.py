"""Experiment registry."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import ExperimentSpec
from .xg_velocity import get_experiment as get_xg_velocity_experiment


ExperimentBuilder = Callable[[Path | None], ExperimentSpec]

_EXPERIMENTS: dict[str, ExperimentBuilder] = {
    "xg_velocity": get_xg_velocity_experiment,
}


def list_experiments() -> tuple[str, ...]:
    return tuple(sorted(_EXPERIMENTS))


def get_experiment(name: str, project_root: Path | None = None) -> ExperimentSpec:
    try:
        builder = _EXPERIMENTS[name]
    except KeyError as exc:
        available = ", ".join(list_experiments())
        raise KeyError(f"Unknown experiment '{name}'. Available: {available}") from exc
    return builder(project_root)


__all__ = [
    "ExperimentSpec",
    "get_experiment",
    "list_experiments",
]
