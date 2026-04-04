#!/usr/bin/env python3
"""
Bittle policy visualization script.

Loads a trained ONNX policy, runs it in the Bittle Brax environment,
and writes MP4 and GIF videos to the project outputs folder.

Usage:
    python Scripts/visualize.py
    python Scripts/visualize.py /path/to/policy.onnx
"""

import os
import sys
import platform
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# MuJoCo backend setup
# ---------------------------------------------------------------------------


def setup_mujoco_backend():
    """Set up a MuJoCo rendering backend based on the current platform."""
    if "MUJOCO_GL" in os.environ:
        del os.environ["MUJOCO_GL"]

    system = platform.system()
    if system == "Darwin":
        backends_to_try = ["glfw"]
    elif system == "Linux":
        backends_to_try = ["egl", "glfw"]
    else:
        backends_to_try = ["glfw"]

    last_error: Optional[Exception] = None
    for backend in backends_to_try:
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco
            print(f"Successfully initialized MuJoCo with {backend} backend")
            return mujoco
        except RuntimeError as exc:
            last_error = exc
            if "invalid value" in str(exc):
                continue
            raise

    raise RuntimeError(
        f"Could not initialize MuJoCo with any backend. Tried: {backends_to_try}. "
        f"Platform: {system}. Last error: {last_error}"
    )


mujoco = setup_mujoco_backend()

import cv2
import jax
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from brax import envs

from locomotion.bittle_env import BittleEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_POLICY_CANDIDATES = [
    REPO_ROOT / "outputs" / "bittle_train_latest" / "policy.onnx",
    REPO_ROOT / "outputs" / "bittle_test_latest" / "policy.onnx",
    REPO_ROOT / "locomotion" / "sim-outputs" / "policies" / "policy.onnx",  # legacy fallback
]

DEFAULT_SCENE_PATH = (
    REPO_ROOT / "assets" / "descriptions" / "bittle" / "mjcf" / "bittle_adapted_scene.xml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "visualize"

DURATION = 5.0
FPS = 50
NUM_STEPS = int(DURATION * FPS)

RENDER_WIDTH = 640
RENDER_HEIGHT = 480


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_policy_path(cli_policy_path: Optional[str] = None) -> Path:
    """Resolve the ONNX policy path from CLI input or known default locations."""
    if cli_policy_path:
        candidate = Path(cli_policy_path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    for candidate in DEFAULT_POLICY_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_POLICY_CANDIDATES[0]



def setup_environment(scene_path: Path) -> Any:
    """Register and create the Bittle environment."""
    print("      Registering environment...")
    envs.register_environment("bittle", BittleEnv)

    print("      Creating environment instance...")
    env = envs.get_environment("bittle", xml_path=str(scene_path))
    return env



def convert_model_ir_version(
    input_path: Path,
    output_path: Path,
    target_ir_version: int = 9,
) -> Path:
    """Create a lower-IR-version ONNX copy for older ONNX Runtime builds."""
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



def load_onnx_policy(policy_path: Path) -> Tuple[ort.InferenceSession, str, str]:
    """Load an ONNX policy and fall back to IR conversion if needed."""
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
        converted_path = convert_model_ir_version(policy_path, converted_path, target_ir_version=9)
        session = ort.InferenceSession(
            str(converted_path),
            providers=["CPUExecutionProvider"],
        )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"      Model loaded (input: {input_name}, output: {output_name})")
    return session, input_name, output_name



def create_inference_fn(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
):
    """Wrap an ONNX Runtime session in a Brax-style inference function."""

    def inference_fn(obs, rng_key):
        del rng_key
        obs_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action = session.run([output_name], {input_name: obs_np})[0]
        return action.flatten(), {}

    return inference_fn



def generate_rollout(env: Any, inference_fn: Any, num_steps: int = NUM_STEPS) -> List[Any]:
    """Generate a rollout by running the policy in the environment."""
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
    env: Any,
    rollout: List[Any],
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> List[np.ndarray]:
    """Render frames from the rollout pipeline states."""
    print(f"      Rendering {len(rollout)} frames at {width}x{height}...")

    mj_model = env.sys.mj_model
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    frames: List[np.ndarray] = []
    for i, pipeline_state in enumerate(rollout):
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.asarray(pipeline_state.q)
        mj_data.qvel[:] = np.asarray(pipeline_state.qd)

        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

        if (i + 1) % 50 == 0:
            print(f"      Progress: {i + 1}/{len(rollout)} frames")

    renderer.close()
    return frames



def save_video_mp4(frames: List[np.ndarray], output_path: Path, fps: int = FPS) -> None:
    """Save frames as MP4 using OpenCV."""
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



def save_video_gif(frames: List[np.ndarray], output_path: Path, fps: int = FPS) -> None:
    """Save frames as an animated GIF using Pillow."""
    if not frames:
        raise ValueError("No frames to write")

    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / fps)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        cli_policy_path = sys.argv[1] if len(sys.argv) > 1 else None
        policy_path = resolve_policy_path(cli_policy_path)
        scene_path = DEFAULT_SCENE_PATH
        output_dir = DEFAULT_OUTPUT_DIR

        print("=" * 60)
        print("BITTLE POLICY VISUALIZATION")
        print("=" * 60)

        print("\nValidating inputs...")
        if not policy_path.exists():
            print(f"Error: Policy file not found: {policy_path}")
            print("Checked these default locations:")
            for candidate in DEFAULT_POLICY_CANDIDATES:
                print(f"  {candidate}")
            print("\nTrain first with something like: python locomotion/train.py --test")
            return 1

        if not scene_path.exists():
            print(f"Error: Scene file not found: {scene_path}")
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nConfiguration:")
        print(f"  Repo root:  {REPO_ROOT}")
        print(f"  Policy:     {policy_path}")
        print(f"  Scene:      {scene_path}")
        print(f"  Output:     {output_dir}")
        print(f"  Duration:   {DURATION}s @ {FPS} FPS ({NUM_STEPS} steps)")
        print(f"  Rendering:  {RENDER_WIDTH}x{RENDER_HEIGHT}")
        print(f"  Backend:    {os.environ.get('MUJOCO_GL', 'auto-detected')}")

        print("\n[1/6] Setting up environment...")
        env = setup_environment(scene_path)
        print(f"      Environment created (obs={env.observation_size}, act={env.action_size})")

        print("[2/6] Loading policy...")
        session, input_name, output_name = load_onnx_policy(policy_path)
        print("      Policy loaded successfully")

        print("[3/6] Creating inference function...")
        inference_fn = create_inference_fn(session, input_name, output_name)
        print("      Inference function ready")

        print("[4/6] Generating rollout...")
        rollout = generate_rollout(env, inference_fn, NUM_STEPS)

        print("[5/6] Rendering frames...")
        frames = render_frames(env, rollout, RENDER_WIDTH, RENDER_HEIGHT)
        print(f"      Rendered {len(frames)} frames successfully")

        print("[6/6] Saving videos...")
        mp4_path = output_dir / "video.mp4"
        gif_path = output_dir / "video.gif"

        print("      Writing MP4...")
        save_video_mp4(frames, mp4_path, FPS)
        print(f"      MP4: {mp4_path}")

        print("      Writing GIF...")
        save_video_gif(frames, gif_path, FPS)
        print(f"      GIF: {gif_path}")

        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETE")
        print("=" * 60)
        print("\nOutputs:")
        print(f"  {mp4_path}")
        print(f"  {gif_path}")
        return 0

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
