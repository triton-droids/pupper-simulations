from brax import envs

from bittle_env import BittleEnv
from training_config import TrainingConfig


def test_train_env_registration_and_setup(xml_path):
    envs.register_environment("bittle", BittleEnv)
    env = envs.get_environment("bittle", xml_path=xml_path)

    assert env.sys.nu == 9

    config = TrainingConfig(test_mode=True)
    d = config.to_dict()

    required_keys = {
        "test_mode", "num_timesteps", "num_evals", "episode_length",
        "num_envs", "batch_size", "unroll_length",
        "num_minibatches", "num_updates_per_batch",
    }
    assert required_keys.issubset(d.keys())
