#!/usr/bin/env python3
"""
Visualize a trained Bittle ONNX policy.

High-level workflow
-------------------
This script takes a trained ONNX policy, runs it inside the Brax Bittle
environment, renders the resulting MuJoCo rollout, and saves both MP4 and GIF
outputs.

The flow is intentionally split into small steps:

1. Resolve CLI inputs and choose sensible defaults.
2. Validate that the policy and scene files exist.
3. Initialize MuJoCo with a backend that works on the current platform.
4. Build the Brax environment and load the ONNX model.
5. Roll out the policy for a fixed number of steps.
6. Render frames and save them to disk.

Keeping those responsibilities separate makes the script easier to debug and
much easier to read than one long monolithic ``main()`` function.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.paths import DEFAULT_SCENE_PATH


DEFAULT_POLICY_CANDIDATES = [
    REPO_ROOT / "outputs" / "bittle_train_latest" / "policy.onnx",
    REPO_ROOT / "outputs" / "bittle_test_latest" / "policy.onnx",
    REPO_ROOT / "locomotion" / "sim-outputs" / "policies" / "policy.onnx",
]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "visualize"


@dataclass(slots=True, frozen=True)
class VisualizationConfig:
    """Resolved runtime configuration for one visualization run."""

    policy_path: Path
    scene_path: Path
    output_dir: Path
    duration_seconds: float
    fps: int
    width: int
    height: int

    @property
    def num_steps(self) -> int:
        """Return the number of environment steps to simulate."""
        return int(self.duration_seconds * self.fps)


@dataclass(slots=True, frozen=True)
class LoadedPolicy:
    """Bundle the ONNX Runtime session with its I/O tensor names."""

    session: Any
    input_name: str
    output_name: str


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Render a rollout video for a trained Bittle ONNX policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "policy_path",
        nargs="?",
        default=None,
        help="Optional path to a specific ONNX policy file.",
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="MuJoCo scene XML to load for visualization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where MP4 and GIF outputs will be written.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Length of the rollout to render, in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Video frame rate and rollout control frequency.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Rendered frame width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Rendered frame height in pixels.",
    )
    return parser


def resolve_policy_path(cli_policy_path: str | None) -> Path:
    """Resolve the explicit policy path or fall back to known default locations."""
    if cli_policy_path:
        candidate = Path(cli_policy_path).expanduser()
        return candidate.resolve() if not candidate.is_absolute() else candidate

    for candidate in DEFAULT_POLICY_CANDIDATES:
        if candidate.exists():
            return candidate

    # Returning the first candidate keeps the eventual error message concrete.
    return DEFAULT_POLICY_CANDIDATES[0]


def build_config(args: argparse.Namespace) -> VisualizationConfig:
    """Convert raw CLI arguments into a fully resolved config object."""
    policy_path = resolve_policy_path(args.policy_path)
    scene_path = args.scene_path.expanduser()
    output_dir = args.output_dir.expanduser()
    return VisualizationConfig(
        policy_path=policy_path,
        scene_path=scene_path,
        output_dir=output_dir,
        duration_seconds=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


def validate_inputs(config: VisualizationConfig) -> None:
    """Raise a clear error if the expected input files do not exist."""
    if not config.policy_path.exists():
        checked_locations = "\n".join(f"  {candidate}" for candidate in DEFAULT_POLICY_CANDIDATES)
        raise FileNotFoundError(
            f"Policy file not found: {config.policy_path}\n"
            f"Checked these default locations:\n{checked_locations}\n\n"
            "Train first with something like: python locomotion/train.py --test"
        )

    if not config.scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {config.scene_path}")


def choose_backend_candidates() -> list[str]:
    """Return MuJoCo backends to try in order of preference."""
    preferred = os.environ.get("MUJOCO_GL")
    system = platform.system()

    if system == "Darwin":
        defaults = ["glfw"]
    elif system == "Linux":
        defaults = ["egl", "glfw"]
    else:
        defaults = ["glfw"]

    if preferred and preferred not in defaults:
        return [preferred, *defaults]
    if preferred:
        return [preferred, *[backend for backend in defaults if backend != preferred]]
    return defaults


def initialize_mujoco_module():
    """Import MuJoCo using the first backend that initializes successfully."""
    candidates = choose_backend_candidates()
    last_error: Exception | None = None

    for backend in candidates:
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco  # Imported lazily so ``--help`` stays lightweight.

            print(f"Initialized MuJoCo with {backend} backend")
            return mujoco
        except RuntimeError as exc:
            last_error = exc
            if "invalid value" in str(exc):
                continue
            raise

    raise RuntimeError(
        f"Could not initialize MuJoCo with any backend. Tried: {candidates}. "
        f"Platform: {platform.system()}. Last error: {last_error}"
    )


def setup_environment(scene_path: Path) -> Any:
    """Register and instantiate the Bittle Brax environment."""
    from brax import envs

    from locomotion.bittle_env import BittleEnv

    print("      Registering environment...")
    envs.register_environment("bittle", BittleEnv)

    print("      Creating environment instance...")
    return envs.get_environment("bittle", xml_path=str(scene_path))


def convert_model_ir_version(
    input_path: Path,
    output_path: Path,
    target_ir_version: int = 9,
) -> Path:
    """
    Create a lower-IR-version ONNX copy for older ONNX Runtime builds.

    Some ONNX Runtime environments lag behind newer ONNX IR versions. Rather
    than failing outright, we create a compatibility copy alongside the
    original model and retry loading from that copy.
    """
    import onnx

    print(f"      Converting ONNX IR version for compatibility: {input_path}")
    model = onnx.load(str(input_path))
    print(f"      Original IR version: {model.ir_version}")

    model.ir_version = target_ir_version
    print(f"      Updated IR version:  {model.ir_version}")

    try:
        onnx.checker.check_model(model)
        print("      Converted model validation successful")
    except Exception as exc:
        print(f"      Warning: converted model validation failed: {exc}")
        print("      Proceeding anyway...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"      Saved converted model to {output_path}")
    return output_path


def load_onnx_policy(policy_path: Path) -> LoadedPolicy:
    """Load an ONNX policy and fall back to an IR-version conversion if needed."""
    import onnxruntime as ort

    print(f"      Loading ONNX model from {policy_path}...")

    try:
        session = ort.InferenceSession(
            str(policy_path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        if "Unsupported model IR version" not in str(exc):
            raise

        converted_path = policy_path.with_name(f"{policy_path.stem}_ir9{policy_path.suffix}")
        converted_path = convert_model_ir_version(
            policy_path,
            converted_path,
            target_ir_version=9,
        )
        session = ort.InferenceSession(
            str(converted_path),
            providers=["CPUExecutionProvider"],
        )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"      Model loaded (input: {input_name}, output: {output_name})")
    return LoadedPolicy(session=session, input_name=input_name, output_name=output_name)


def create_inference_fn(loaded_policy: LoadedPolicy):
    """Wrap ONNX Runtime in the callable shape expected by the rollout code."""

    def inference_fn(obs: Any, rng_key: Any):
        del rng_key
        obs_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action = loaded_policy.session.run(
            [loaded_policy.output_name],
            {loaded_policy.input_name: obs_np},
        )[0]
        return action.flatten(), {}

    return inference_fn


def generate_rollout(env: Any, inference_fn: Any, num_steps: int) -> list[Any]:
    """Run the policy in the environment and collect pipeline states."""
    import jax

    print(f"      Running {num_steps} steps...")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for _ in range(num_steps):
        rng, key_sample = jax.random.split(rng)
        action, _ = inference_fn(state.obs, key_sample)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)

    print(f"      Collected {len(rollout)} states")
    return rollout


def render_frames(
    *,
    mujoco_module: Any,
    env: Any,
    rollout: list[Any],
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Render MuJoCo frames from the collected rollout states."""
    print(f"      Rendering {len(rollout)} frames at {width}x{height}...")

    mj_model = env.sys.mj_model
    renderer = mujoco_module.Renderer(mj_model, height=height, width=width)

    frames: list[np.ndarray] = []
    for index, pipeline_state in enumerate(rollout, start=1):
        mj_data = mujoco_module.MjData(mj_model)
        mj_data.qpos[:] = np.asarray(pipeline_state.q)
        mj_data.qvel[:] = np.asarray(pipeline_state.qd)

        mujoco_module.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

        if index % 50 == 0:
            print(f"      Progress: {index}/{len(rollout)} frames")

    renderer.close()
    return frames


def save_video_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Save rendered frames as an MP4 using OpenCV."""
    import cv2

    if not frames:
        raise ValueError("No frames to write")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()


def save_video_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Save rendered frames as an animated GIF using Pillow."""
    from PIL import Image

    if not frames:
        raise ValueError("No frames to write")

    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def print_banner(title: str) -> None:
    """Print a section banner for the CLI output."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_configuration(config: VisualizationConfig) -> None:
    """Print the resolved runtime configuration."""
    print("\nConfiguration:")
    print(f"  Repo root:  {REPO_ROOT}")
    print(f"  Policy:     {config.policy_path}")
    print(f"  Scene:      {config.scene_path}")
    print(f"  Output:     {config.output_dir}")
    print(f"  Duration:   {config.duration_seconds}s @ {config.fps} FPS ({config.num_steps} steps)")
    print(f"  Rendering:  {config.width}x{config.height}")
    print(f"  Backend:    {os.environ.get('MUJOCO_GL', 'auto-detected')}")


def main() -> int:
    """Run the end-to-end policy visualization workflow."""
    args = build_argparser().parse_args()
    config = build_config(args)

    try:
        print_banner("BITTLE POLICY VISUALIZATION")

        print("\nValidating inputs...")
        validate_inputs(config)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        mujoco_module = initialize_mujoco_module()
        print_configuration(config)

        print("\n[1/6] Setting up environment...")
        env = setup_environment(config.scene_path)
        print(f"      Environment created (obs={env.observation_size}, act={env.action_size})")

        print("[2/6] Loading policy...")
        loaded_policy = load_onnx_policy(config.policy_path)
        print("      Policy loaded successfully")

        print("[3/6] Creating inference function...")
        inference_fn = create_inference_fn(loaded_policy)
        print("      Inference function ready")

        print("[4/6] Generating rollout...")
        rollout = generate_rollout(env, inference_fn, config.num_steps)

        print("[5/6] Rendering frames...")
        frames = render_frames(
            mujoco_module=mujoco_module,
            env=env,
            rollout=rollout,
            width=config.width,
            height=config.height,
        )
        print(f"      Rendered {len(frames)} frames successfully")

        print("[6/6] Saving videos...")
        mp4_path = config.output_dir / "video.mp4"
        gif_path = config.output_dir / "video.gif"

        print("      Writing MP4...")
        save_video_mp4(frames, mp4_path, config.fps)
        print(f"      MP4: {mp4_path}")

        print("      Writing GIF...")
        save_video_gif(frames, gif_path, config.fps)
        print(f"      GIF: {gif_path}")

        print("")
        print_banner("VISUALIZATION COMPLETE")
        print("\nOutputs:")
        print(f"  {mp4_path}")
        print(f"  {gif_path}")
        return 0

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
