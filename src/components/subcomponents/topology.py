"""Robot topology: name → body/site ID resolution."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco


class TopologySubComponent:
    """Resolve robot body/site/geom IDs from string names.

    All lookups happen once at construction time. Missing bodies fall back
    to index 0 (with a warning) so the env still runs in degraded mode.
    """

    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        foot_body_names: tuple[str, ...],
        foot_site_names: tuple[str, ...] = (),
        nonfoot_body_names: tuple[str, ...] = (),
        illegal_contact_force_threshold: float = 10.0,
    ):
        self.mj_model = mj_model
        self.illegal_contact_force_threshold = illegal_contact_force_threshold

        self._feet_body_ids = self._resolve_bodies(foot_body_names, expected=4)
        self._feet_site_ids = self._resolve_sites(foot_site_names)
        self._nonfoot_body_ids = self._resolve_bodies(nonfoot_body_names, expected=0)
        self._penalized_body_ids = self._resolve_bodies(
            ("torso",) + tuple(foot_body_names), expected=0
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def feet_body_ids(self) -> jax.Array:
        return self._feet_body_ids

    @property
    def feet_site_ids(self) -> jax.Array:
        return self._feet_site_ids

    @property
    def penalized_body_ids(self) -> jax.Array:
        return self._penalized_body_ids

    @property
    def nonfoot_body_ids(self) -> jax.Array:
        return self._nonfoot_body_ids

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_bodies(self, names: tuple[str, ...], expected: int) -> jax.Array:
        ids = []
        for name in names:
            idx = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, name
            )
            if idx >= 0:
                ids.append(idx)
            else:
                print(f"  [TopologySubComponent] Warning: body '{name}' not found!")
        if expected and len(ids) == expected:
            return jp.array(ids)
        if not ids:
            return jp.array([0])
        return jp.array(ids)

    def _resolve_sites(self, names: tuple[str, ...]) -> jax.Array:
        ids = []
        for name in names:
            idx = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, name
            )
            if idx >= 0:
                ids.append(idx)
        if not ids:
            return jp.array([0, 0, 0, 0])
        return jp.array(ids)


__all__ = ["TopologySubComponent"]
