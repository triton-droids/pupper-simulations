"""
Domain-randomization helpers for sim-to-real training experiments.

The current randomization policy perturbs:

- surface friction
- actuator gains
- corresponding actuator bias terms

The function returns both a randomized system tree and the ``in_axes`` tree
that Brax expects when vectorizing randomization across environments.
"""

from __future__ import annotations

from typing import Any

import jax


FRICTION_RANGE = (0.6, 1.4)
ACTUATOR_GAIN_DELTA_RANGE = (-5.0, 5.0)


def _randomize_single_system(sys: Any, rng: jax.Array):
    """Randomize one system instance for one environment sample."""
    friction_key, gain_key = jax.random.split(rng)

    friction_scale = jax.random.uniform(
        friction_key,
        (1,),
        minval=FRICTION_RANGE[0],
        maxval=FRICTION_RANGE[1],
    )
    randomized_friction = sys.geom_friction.at[:, 0].set(friction_scale)

    gain_delta = jax.random.uniform(
        gain_key,
        (1,),
        minval=ACTUATOR_GAIN_DELTA_RANGE[0],
        maxval=ACTUATOR_GAIN_DELTA_RANGE[1],
    )
    randomized_gain = sys.actuator_gainprm.at[:, 0].set(
        sys.actuator_gainprm[:, 0] + gain_delta
    )
    randomized_bias = sys.actuator_biasprm.at[:, 1].set(
        -(sys.actuator_gainprm[:, 0] + gain_delta)
    )

    return randomized_friction, randomized_gain, randomized_bias


def domain_randomize(sys: Any, rng: jax.Array):
    """
    Apply the project's domain-randomization policy to a Brax system tree.

    Returns:
        A tuple ``(randomized_sys, in_axes)`` compatible with Brax PPO's
        ``randomization_fn`` callback API.
    """

    randomize_many = jax.vmap(lambda rng_key: _randomize_single_system(sys, rng_key))
    friction, gain, bias = randomize_many(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    randomized_sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )
    return randomized_sys, in_axes
