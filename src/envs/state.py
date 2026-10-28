"""Shared environment state types."""

from __future__ import annotations

from typing import Any, Mapping, Union

import jax
from flax import struct
from mujoco import mjx

from .info import EnvInfo


Observation = Union[jax.Array, Mapping[str, jax.Array]]


@struct.dataclass
class State:
    """Environment state for training and inference."""

    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    info: EnvInfo


__all__ = ["EnvInfo", "Observation", "State"]
