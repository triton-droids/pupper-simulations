#!/usr/bin/env python3
"""Train Bittle locomotion policy with wandb experiment tracking."""
import os
os.environ["MUJOCO_GL"] = "egl"

import argparse, functools, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")

import wandb
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from bittle_env import BittleEnv
from domain_randomization import domain_randomize
from onnx_export import export_policy_to_onnx
from training_config import TrainingConfig
from video_recorder import record_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", default="../assets/descriptions/bittle/mjcf/bittle_scene.xml")
    parser.add_argument("--output", default="outputs/policy.onnx")
    parser.add_argument("--video-output", default="outputs/videos/latest_video.mp4",
                        help="Path for recorded video (default: outputs/videos/latest_video.mp4)")
    args = parser.parse_args()

    config = TrainingConfig()

    run = wandb.init(project="bittle-locomotion", config=config.to_dict())
    print(f"Training Bittle | wandb run: {run.url}")
    print(f"Config: {config.to_dict()}")

    envs.register_environment("bittle", BittleEnv)
    env = envs.get_environment("bittle", xml_path=args.xml_path)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
    )

    def progress(step, metrics):
        reward = float(metrics.get("eval/episode_reward", 0))
        print(f"  Step {step:>13,} | Reward: {reward:.4f}")
        wandb.log({k: float(v) for k, v in metrics.items()}, step=step)

    make_policy, params, _ = ppo.train(
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
        reward_scaling=config.reward_scaling,
        normalize_observations=config.normalize_observations,
        action_repeat=config.action_repeat,
        discounting=config.discounting,
        learning_rate=config.learning_rate,
        entropy_cost=config.entropy_cost,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize,
        seed=config.seed,
    )

    # Export ONNX locally and to wandb
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_policy_to_onnx(params, args.output)
    print(f"Policy saved to {args.output}")

    artifact = wandb.Artifact("bittle-policy", type="model")
    artifact.add_file(args.output)
    wandb.log_artifact(artifact)

    # Record 5s video and log to wandb
    diagnostics_path = args.video_output.replace(".mp4", "_diagnostics.txt")
    try:
        record_video(env, make_policy, params, args.video_output,
                     diagnostics_path=diagnostics_path)
        wandb.log({"eval_video": wandb.Video(args.video_output, fps=50, format="mp4")})
        wandb.save(diagnostics_path)
    except Exception as e:
        print(f"Warning: Video recording failed: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
