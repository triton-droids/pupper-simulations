"""Tests for locomotion/domain_randomization.py (1 test)."""

import jax
import jax.numpy as jp
import pytest


class _FakeSys:
    """Minimal stand-in for brax.base.System with JAX pytree + tree_replace."""

    def __init__(self, geom_friction, actuator_gainprm, actuator_biasprm):
        self.geom_friction = geom_friction
        self.actuator_gainprm = actuator_gainprm
        self.actuator_biasprm = actuator_biasprm

    def tree_replace(self, replacements):
        new = _FakeSys(self.geom_friction, self.actuator_gainprm, self.actuator_biasprm)
        for k, v in replacements.items():
            setattr(new, k, v)
        return new


# Register as JAX pytree so jax.tree_util.tree_map works
jax.tree_util.register_pytree_node(
    _FakeSys,
    lambda s: ((s.geom_friction, s.actuator_gainprm, s.actuator_biasprm), None),
    lambda _, xs: _FakeSys(*xs),
)


def test_domain_randomize():
    """domain_randomize returns (sys, in_axes) with correctly shaped randomized fields."""
    from domain_randomization import domain_randomize

    n_geoms = 4
    n_actuators = 9

    sys = _FakeSys(
        geom_friction=jp.ones((n_geoms, 3)),
        actuator_gainprm=jp.zeros((n_actuators, 10)),
        actuator_biasprm=jp.zeros((n_actuators, 10)),
    )

    batch = 4
    rng = jax.random.split(jax.random.PRNGKey(0), batch)

    new_sys, in_axes = domain_randomize(sys, rng)

    # Shapes should have batch dimension prepended
    assert new_sys.geom_friction.shape == (batch, n_geoms, 3)
    assert new_sys.actuator_gainprm.shape == (batch, n_actuators, 10)
    assert new_sys.actuator_biasprm.shape == (batch, n_actuators, 10)

    # in_axes should mark randomized fields with 0, others with None
    assert in_axes.geom_friction == 0
    assert in_axes.actuator_gainprm == 0
    assert in_axes.actuator_biasprm == 0
