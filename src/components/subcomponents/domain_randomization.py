"""Domain randomization sub-component."""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx


class DomainRandomizationSubComponent:
    """Model-parameter randomization sub-component."""

    def __init__(self, *, feet_geom_ids: jax.Array | None = None):
        self.feet_geom_ids = feet_geom_ids if feet_geom_ids is not None else jp.array([], dtype=jp.int32)

    def randomize_model(self, model: mjx.Model, *,
                        friction_coeff: jax.Array,
                        inertia_factors: jax.Array,
                        restitution_coeff: jax.Array,
                        feet_friction: jax.Array,
                        mass_factors: jax.Array,
                        com_offset: jax.Array) -> mjx.Model:
        """Apply model-parameter randomization."""
        return randomize_model_params(
            model=model,
            friction_coeff=friction_coeff,
            inertia_factors=inertia_factors,
            restitution_coeff=restitution_coeff,
            feet_friction=feet_friction,
            feet_geom_ids=self.feet_geom_ids,
            mass_factors=mass_factors,
            com_offset=com_offset,
        )


def randomize_model_params(
    model: mjx.Model,
    friction_coeff: jax.Array,
    inertia_factors: jax.Array,
    restitution_coeff: jax.Array,
    feet_friction: jax.Array,
    feet_geom_ids: jax.Array,
    mass_factors: jax.Array,
    com_offset: jax.Array,
) -> mjx.Model:
    """Apply currently enabled MJX model-parameter randomization."""
    safe_friction = jp.maximum(friction_coeff, 0.01)
    new_geom_friction = model.geom_friction.at[:, 0].set(safe_friction)

    safe_feet_friction = jp.maximum(feet_friction, 0.01)
    if feet_geom_ids.shape[0] > 0:
        new_geom_friction = jax.lax.fori_loop(
            0,
            feet_geom_ids.shape[0],
            lambda i, arr: arr.at[feet_geom_ids[i], 0].set(safe_feet_friction),
            new_geom_friction,
        )

    return model.replace(geom_friction=new_geom_friction)


__all__ = ["DomainRandomizationSubComponent", "randomize_model_params"]
