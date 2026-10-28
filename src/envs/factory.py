"""Environment factory and registry."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from cfg.experiments import ExperimentSpec, get_experiment


def load_symbol(symbol_path: str) -> Any:
    """Resolve a ``"module:symbol"`` path to the live object."""
    module_name, symbol_name = symbol_path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, symbol_name)


# Kept for in-package call sites that imported the underscored name.
_load_symbol = load_symbol


def build_env(experiment: str | ExperimentSpec = "xg_velocity", **overrides: Any) -> Any:
    """Create an environment from an experiment name or descriptor."""
    if isinstance(experiment, str):
        spec = get_experiment(experiment, Path(__file__).resolve().parents[2])
    else:
        spec = experiment

    config = spec.build_env_config()
    for dotted_key, value in overrides.items():
        target = config
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)

    env_cls = load_symbol(spec.env_class_path)
    env = env_cls(config=config, component_name=spec.component_name)
    env.experiment_spec = spec
    return env
