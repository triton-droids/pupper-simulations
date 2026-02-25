from training_config import TrainingConfig


def test_training_config():
    # Test mode
    test_cfg = TrainingConfig(test_mode=True)
    assert test_cfg.test_mode is True
    assert test_cfg.num_timesteps == 10_000
    assert test_cfg.num_evals == 2
    assert test_cfg.episode_length == 100
    assert test_cfg.num_envs == 8
    assert test_cfg.batch_size == 128
    assert test_cfg.unroll_length == 5
    assert test_cfg.num_minibatches == 2
    assert test_cfg.num_updates_per_batch == 1

    # Full mode
    full_cfg = TrainingConfig(test_mode=False)
    assert full_cfg.test_mode is False
    assert full_cfg.num_timesteps == 10_000_000
    assert full_cfg.num_evals == 10
    assert full_cfg.episode_length == 1000
    assert full_cfg.num_envs == 4096
    assert full_cfg.batch_size == 512
    assert full_cfg.unroll_length == 20
    assert full_cfg.num_minibatches == 8
    assert full_cfg.num_updates_per_batch == 1

    # to_dict returns all 9 fields with correct types
    d = test_cfg.to_dict()
    assert len(d) == 9
    assert isinstance(d["test_mode"], bool)
    assert isinstance(d["num_timesteps"], int)
    assert isinstance(d["num_evals"], int)
    assert isinstance(d["episode_length"], int)
    assert isinstance(d["num_envs"], int)
    assert isinstance(d["batch_size"], int)
    assert isinstance(d["unroll_length"], int)
    assert isinstance(d["num_minibatches"], int)
    assert isinstance(d["num_updates_per_batch"], int)
