#!/usr/bin/env python3
"""
Bittle Policy Visualization Script

Loads a trained policy from ONNX format and visualizes it in the Bittle simulation
environment. Outputs both MP4 and GIF videos.

Usage:
    python visualize.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Any

# Set MuJoCo backend BEFORE importing MuJoCo/Brax
# osmesa enables CPU-based headless rendering (works everywhere, no GPU needed)
os.environ["MUJOCO_GL"] = "osmesa"

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import cv2
from PIL import Image
import onnxruntime as ort

from brax import envs

# Add locomotion directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "locomotion"))
from bittle_env import BittleEnv


# ============================================================================
# Configuration
# ============================================================================

POLICY_PATH = "locomotion/sim-outputs/policies/policy.onnx"
SCENE_PATH = "locomotion/bittle_adapted_scene.xml"
OUTPUT_DIR = "outputs"

DURATION = 5.0  # seconds
FPS = 50
NUM_STEPS = 250  # 5 seconds at 50Hz

RENDER_WIDTH = 640
RENDER_HEIGHT = 480


# ============================================================================
# Helper Functions
# ============================================================================

def setup_environment(scene_path: str) -> Any:
    """
    Register and create the Bittle environment.

    Args:
        scene_path: Path to the scene XML file

    Returns:
        Configured Brax environment
    """
    print("      Registering environment...")
    envs.register_environment("bittle", BittleEnv)

    print("      Creating environment instance...")
    env = envs.get_environment("bittle", xml_path=scene_path)

    return env


def load_onnx_policy(policy_path: str) -> Tuple[ort.InferenceSession, str, str]:
    """
    Load ONNX policy model.

    Args:
        policy_path: Path to ONNX policy file

    Returns:
        Tuple of (session, input_name, output_name)
    """
    print(f"      Loading ONNX model from {policy_path}...")

    # Load ONNX model with CPU execution provider
    session = ort.InferenceSession(
        policy_path,
        providers=['CPUExecutionProvider']
    )

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"      Model loaded (input: {input_name}, output: {output_name})")

    return session, input_name, output_name


def create_inference_fn(session: ort.InferenceSession,
                       input_name: str,
                       output_name: str):
    """
    Create inference function from ONNX session.

    Args:
        session: ONNX inference session
        input_name: Name of input tensor
        output_name: Name of output tensor

    Returns:
        Inference function compatible with Brax
    """
    def inference_fn(obs, rng_key):
        """Run inference using ONNX model."""
        # Convert JAX array to numpy and reshape for batch dimension
        obs_np = np.array(obs).reshape(1, -1).astype(np.float32)

        # Run ONNX inference
        action = session.run([output_name], {input_name: obs_np})[0]

        # Return action and empty dict (matching Brax interface)
        return action.flatten(), {}

    return inference_fn


def generate_rollout(env: Any,
                    inference_fn: Any,
                    num_steps: int = 250) -> List[Any]:
    """
    Generate a rollout by running the policy in the environment.

    Args:
        env: Brax environment
        inference_fn: Policy inference function
        num_steps: Number of steps to simulate

    Returns:
        List of pipeline states
    """
    print(f"      Running {num_steps} steps...")

    # JIT compile for performance
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Reset environment
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    # Run episode
    for _ in range(num_steps):
        rng, key_sample = jax.random.split(rng)
        obs = state.obs
        action, _ = inference_fn(obs, key_sample)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)

    print(f"      Collected {len(rollout)} states")

    return rollout


def render_frames(env: Any,
                 rollout: List[Any],
                 width: int = 640,
                 height: int = 480) -> List[np.ndarray]:
    """
    Render frames from pipeline states.

    Args:
        env: Brax environment
        rollout: List of pipeline states
        width: Frame width
        height: Frame height

    Returns:
        List of RGB frames as numpy arrays
    """
    print(f"      Rendering {len(rollout)} frames at {width}x{height}...")

    # Get MuJoCo model from Brax environment
    mj_model = env.sys.mj_model

    # Create renderer (uses backend set via environment variable)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    frames = []
    for i, pipeline_state in enumerate(rollout):
        # Create MuJoCo data and set state
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(pipeline_state.q)
        mj_data.qvel[:] = np.array(pipeline_state.qd)

        # Forward kinematics
        mujoco.mj_forward(mj_model, mj_data)

        # Render frame
        renderer.update_scene(mj_data)
        pixels = renderer.render()  # Returns RGB numpy array
        frames.append(pixels)

        # Progress indicator every 50 frames
        if (i + 1) % 50 == 0:
            print(f"      Progress: {i + 1}/{len(rollout)} frames")

    renderer.close()

    return frames


def save_video_mp4(frames: List[np.ndarray],
                  output_path: Path,
                  fps: int = 50) -> None:
    """
    Save frames as MP4 video using OpenCV.

    Args:
        frames: List of RGB frames
        output_path: Path to save MP4
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames to write")

    height, width, _ = frames[0].shape

    # Create VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    # Write frames (convert RGB to BGR for OpenCV)
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def save_video_gif(frames: List[np.ndarray],
                  output_path: Path,
                  fps: int = 50) -> None:
    """
    Save frames as animated GIF using Pillow.

    Args:
        frames: List of RGB frames
        output_path: Path to save GIF
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames to write")

    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Calculate duration per frame in milliseconds
    duration_ms = int(1000 / fps)

    # Save as animated GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0  # Loop forever
    )


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    try:
        print("=" * 60)
        print("BITTLE POLICY VISUALIZATION")
        print("=" * 60)

        # Setup paths
        policy_path = Path(POLICY_PATH)
        scene_path = Path(SCENE_PATH)
        output_dir = Path(OUTPUT_DIR)

        # Validate inputs
        print("\nValidating inputs...")
        if not policy_path.exists():
            print(f"Error: Policy file not found: {policy_path}")
            print("Please train and export a policy first by running: python locomotion/train.py")
            return 1

        if not scene_path.exists():
            print(f"Error: Scene file not found: {scene_path}")
            return 1

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nConfiguration:")
        print(f"  Policy:    {policy_path}")
        print(f"  Scene:     {scene_path}")
        print(f"  Output:    {output_dir}/")
        print(f"  Duration:  {DURATION}s @ {FPS} FPS ({NUM_STEPS} steps)")
        print(f"  Rendering: {RENDER_WIDTH}x{RENDER_HEIGHT}")
        print(f"  Backend:   {os.environ['MUJOCO_GL']}")

        # Step 1: Setup environment
        print("\n[1/6] Setting up environment...")
        env = setup_environment(str(scene_path))
        print(f"      Environment created (obs={env.observation_size}, act={env.action_size})")

        # Step 2: Load policy
        print("[2/6] Loading policy...")
        session, input_name, output_name = load_onnx_policy(str(policy_path))
        print("      Policy loaded successfully")

        # Step 3: Create inference function
        print("[3/6] Creating inference function...")
        inference_fn = create_inference_fn(session, input_name, output_name)
        print("      Inference function ready")

        # Step 4: Generate rollout
        print("[4/6] Generating rollout...")
        rollout = generate_rollout(env, inference_fn, NUM_STEPS)

        # Step 5: Render frames
        print("[5/6] Rendering frames...")
        frames = render_frames(env, rollout, RENDER_WIDTH, RENDER_HEIGHT)
        print(f"      Rendered {len(frames)} frames successfully")

        # Step 6: Save videos
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
        print(f"\nOutputs:")
        print(f"  {mp4_path}")
        print(f"  {gif_path}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
