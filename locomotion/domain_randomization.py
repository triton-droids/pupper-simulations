"""
Make each simulated robot world a little different from the others.

Why this file exists
--------------------
When every training episode uses the exact same friction and motor strength,
the learned policy can become fragile and overfit to that one perfect virtual
world. Domain randomization deliberately adds small variations so the policy
has to succeed across a range of conditions and is less likely to fall apart
when the real robot or a new scene is not identical to training.

What gets changed right now
---------------------------
- floor/foot friction
- motor gain strength
- the matching motor bias term that keeps the actuator math consistent

The main function returns two things:
- a batched copy of the system with per-environment random values baked in
- an ``in_axes`` description telling Brax which pieces vary across the batch
"""

from __future__ import annotations

from typing import Any

import jax


FRICTION_RANGE = (0.6, 1.4)
ACTUATOR_GAIN_DELTA_RANGE = (-5.0, 5.0)


def _randomize_single_system(sys: Any, rng: jax.Array):
    """
    Create one slightly altered copy of the simulator settings.

    In everyday terms, this is the "make one practice world" helper. It picks
    one random friction value and one random motor-strength adjustment, then
    builds the matching actuator settings for that world.
    """
    friction_key, gain_key = jax.random.split(rng)

    # Pick one shared friction multiplier for this environment sample so the
    # whole robot experiences a slightly grippier or slipperier world.
    friction_scale = jax.random.uniform(
        friction_key,
        (1,),
        minval=FRICTION_RANGE[0],
        maxval=FRICTION_RANGE[1],
    )
    randomized_friction = sys.geom_friction.at[:, 0].set(friction_scale)

    # Pick one motor-gain offset for this sample. This simulates actuators that
    # are a little stronger or weaker than the nominal model says.
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

    This function is the batch wrapper around ``_randomize_single_system``.
    Brax may be training many environments in parallel, so we need one random
    simulator variant per environment. ``vmap`` handles that by calling the
    single-environment helper once per RNG key.

    Returns:
        A tuple ``(randomized_sys, in_axes)`` compatible with Brax PPO's
        ``randomization_fn`` callback API.
    """

    # Build one randomized set of simulator parameters per environment in the
    # batch, all starting from the same base MuJoCo system definition.
    randomize_many = jax.vmap(lambda rng_key: _randomize_single_system(sys, rng_key))
    friction, gain, bias = randomize_many(rng)

    # Tell Brax which system fields now have a batch dimension. Most of the
    # simulator tree stays shared; only these randomized arrays vary.
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    # Drop the randomized values back into a batched system tree that PPO can
    # hand to many environments at once.
    randomized_sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )
    return randomized_sys, in_axes
