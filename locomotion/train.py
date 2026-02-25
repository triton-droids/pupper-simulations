#!/usr/bin/env python3
"""Train Bittle locomotion policy."""
import os
os.environ["MUJOCO_GL"] = "egl"

import argparse, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")

from brax import envs
from brax.training.agents.ppo import train as ppo
from bittle_env import BittleEnv
from onnx_export import export_policy_to_onnx
from training_config import TrainingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--xml-path", default="bittle_adapted_scene.xml")
    parser.add_argument("--output", default="outputs/policy.onnx")
    args = parser.parse_args()

    config = TrainingConfig(test_mode=args.test)
    mode = "TEST" if args.test else "TRAIN"
    print(f"Training Bittle ({mode} mode) | {config.to_dict()}")

    envs.register_environment("bittle", BittleEnv)
    env = envs.get_environment("bittle", xml_path=args.xml_path)

    def progress(step, metrics):
        print(f"  Step {step:>10,} | Reward: {float(metrics['eval/episode_reward']):.4f}")

    _, params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        num_timesteps=config.num_timesteps,
        num_evals=config.num_evals,
        episode_length=config.episode_length,
        num_envs=config.num_envs,
        batch_size=config.batch_size,
        unroll_length=config.unroll_length,
        num_minibatches=config.num_minibatches,
        num_updates_per_batch=config.num_updates_per_batch,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_policy_to_onnx(params, args.output)
    print(f"Policy saved to {args.output}")

if __name__ == "__main__":
    main()
