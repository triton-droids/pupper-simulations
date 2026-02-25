def test_bittle_env_construction(bittle_env):
    assert bittle_env.sys.nu == 9
    assert bittle_env._base_body_id >= 0
    assert bittle_env._default_pose.shape == (9,)
    assert len(bittle_env._lower_leg_body_id) == 4
    assert bittle_env._action_scale == 0.5
    assert bittle_env.backend == "mjx"
