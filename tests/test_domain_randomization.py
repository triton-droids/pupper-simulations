import jax
import jax.numpy as jp


def test_domain_randomize(brax_sys):
    from domain_randomization import domain_randomize

    batch_size = 4
    rng_batch = jax.random.split(jax.random.PRNGKey(0), batch_size)

    rand_sys, in_axes = domain_randomize(brax_sys, rng_batch)

    # Randomized fields should have a leading batch dimension
    assert rand_sys.geom_friction.shape[0] == batch_size
    assert rand_sys.actuator_gainprm.shape[0] == batch_size
    assert rand_sys.actuator_biasprm.shape[0] == batch_size

    # in_axes should mark randomized fields as 0 (vmapped)
    assert in_axes.geom_friction == 0
    assert in_axes.actuator_gainprm == 0
    assert in_axes.actuator_biasprm == 0
