"""Domain-randomization force application sub-component."""

from __future__ import annotations

from typing import Any

import jax.numpy as jp


class PhysicsSubComponent:
    """Applies domain-randomization effects as external forces."""

    def __init__(self, *, domain_rand: Any = None):
        self.domain_rand = domain_rand

    def apply_forces(self, *, model: Any, data: Any,
                     info: dict[str, Any],
                     qfrc_applied: Any = None) -> Any:
        """Apply domain-randomization forces to the generalized force vector."""
        return apply_domain_randomization_forces(
            model=model, data=data, info=info,
            qfrc_applied=qfrc_applied, domain_rand=self.domain_rand,
        )


def apply_domain_randomization_forces(
    *,
    model: Any,
    data: Any,
    info: dict[str, Any],
    qfrc_applied,
    domain_rand: Any,
):
    """Apply DR external-force approximations matching the original env."""
    if qfrc_applied is None:
        qfrc_applied = jp.zeros(model.nv)

    if domain_rand and getattr(domain_rand, "randomize_mass", False):
        mass_factors = info.get("mass_factors", jp.ones(model.nbody))
        avg_mass_factor = jp.mean(mass_factors)
        total_mass = jp.sum(model.body_mass)
        gravity_adjustment = (avg_mass_factor - 1.0) * total_mass * 9.81
        qfrc_applied = qfrc_applied.at[2].add(-gravity_adjustment)

    if domain_rand and getattr(domain_rand, "randomize_com_offset", False):
        com_offset = info.get("com_offset", jp.zeros(3))
        total_mass = jp.sum(model.body_mass)
        gravity_force = jp.array([0.0, 0.0, -total_mass * 9.81])
        com_torque = jp.cross(com_offset, gravity_force)
        qfrc_applied = qfrc_applied.at[3:6].add(com_torque)

    if domain_rand and getattr(domain_rand, "randomize_inertia", False):
        inertia_factors = info.get("inertia_factors", jp.ones((model.nbody, 3)))
        avg_inertia_factor = jp.mean(inertia_factors)
        base_ang_vel = data.qvel[3:6]
        inertia_damping_coeff = 0.5
        inertia_damping = -(avg_inertia_factor - 1.0) * inertia_damping_coeff * base_ang_vel
        qfrc_applied = qfrc_applied.at[3:6].add(inertia_damping)

    if domain_rand and getattr(domain_rand, "randomize_friction", False):
        friction_coeff = info.get("friction_coeff", 1.0)
        friction_damping_coeff = 2.0
        friction_damping = (friction_coeff - 1.0) * friction_damping_coeff * data.qvel[:2]
        qfrc_applied = qfrc_applied.at[0].add(-friction_damping[0])
        qfrc_applied = qfrc_applied.at[1].add(-friction_damping[1])

    return qfrc_applied


__all__ = ["PhysicsSubComponent", "apply_domain_randomization_forces"]
