"""Current XG velocity experiment."""

from __future__ import annotations

from pathlib import Path

from cfg.xg_config import get_xg_config, get_xg_ppo_config
from robots import build_xg_spec
from tasks import build_velocity_task

from .base import ExperimentSpec


def get_experiment(project_root: Path | None = None) -> ExperimentSpec:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    return ExperimentSpec(
        name="xg_velocity",
        robot=build_xg_spec(project_root),
        task=build_velocity_task(),
        env_config_builder=get_xg_config,
        ppo_config_builder=get_xg_ppo_config,
        env_class_path="envs.backends.generic_env:GenericEnv",
        component_name="velocity",
        notes="Phase-one adapter for the existing velocity MJX environment.",
    )
