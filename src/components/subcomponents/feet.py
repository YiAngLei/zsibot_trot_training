"""Foot contact bookkeeping sub-component."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp


class FeetSubComponent:
    """Foot contact state tracking sub-component."""

    def __init__(self, *, feet_body_ids: jax.Array, dt: float):
        self.feet_body_ids = feet_body_ids
        self.dt = dt

    def update(self, *, data: Any, last_contact: jax.Array,
               feet_air_time: jax.Array,
               feet_contact_time: jax.Array) -> dict[str, jax.Array]:
        """Update Isaac Gym-style foot contact filters and timers."""
        return update_feet_contact_state(
            data=data,
            feet_body_ids=self.feet_body_ids,
            last_contact=last_contact,
            feet_air_time=feet_air_time,
            feet_contact_time=feet_contact_time,
            dt=self.dt,
        )


def update_feet_contact_state(
    *,
    data: Any,
    feet_body_ids: jax.Array,
    last_contact: jax.Array,
    feet_air_time: jax.Array,
    feet_contact_time: jax.Array,
    dt: float,
) -> dict[str, jax.Array]:
    """Update Isaac Gym-style foot contact filters and timers."""
    feet_forces = data.cfrc_ext[feet_body_ids, 3:6]
    contact = feet_forces[:, 2] > 1.0
    contact_filt = contact | last_contact
    first_contact = (feet_air_time > 0) * contact_filt

    new_feet_air_time = (feet_air_time + dt) * ~contact_filt
    new_feet_contact_time = (
        feet_contact_time + dt * contact_filt.astype(jp.float32)
    ) * contact_filt.astype(jp.float32)

    return {
        "contact": contact,
        "contact_filt": contact_filt,
        "first_contact": first_contact,
        "feet_air_time": new_feet_air_time,
        "feet_contact_time": new_feet_contact_time,
    }


__all__ = ["FeetSubComponent", "update_feet_contact_state"]
