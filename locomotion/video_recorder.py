"""Video recording utility for trained Brax policies.

Records a short rollout of a trained policy and saves it as MP4.
Follows the same rendering approach as legacy/visualize.py.
"""

import numpy as np
import jax
import jax.numpy as jp
import mujoco
from brax import math


def generate_rollout(env, inference_fn, num_steps=250, seed=0):
    """Run the trained policy in the environment and collect pipeline states.

    Args:
        env: Brax environment instance.
        inference_fn: Policy function taking (obs, rng_key) -> (action, extras).
        num_steps: Number of environment steps to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (rollout, diagnostics) where:
        - rollout: List of pipeline_state objects (length num_steps + 1, including reset).
        - diagnostics: Dict with 'observations', 'actions', 'rewards', 'dones',
          and 'velocities' (local-frame base velocity, shape (num_steps, 3)) arrays.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    observations = []
    actions = []
    rewards = []
    dones = []
    velocities = []

    for _ in range(num_steps):
        rng, key_sample = jax.random.split(rng)
        action, _ = inference_fn(state.obs, key_sample)
        observations.append(np.array(state.obs))
        actions.append(np.array(action))
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)
        rewards.append(np.array(state.reward))
        dones.append(np.array(state.done))

        ps = state.pipeline_state
        world_vel = ps.xd.vel[env._base_body_id]
        inv_rot = math.quat_inv(ps.x.rot[env._base_body_id])
        local_vel = math.rotate(world_vel, inv_rot)
        velocities.append(np.array(local_vel))

    diagnostics = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'velocities': np.array(velocities),
    }
    return rollout, diagnostics


def render_frames(env, rollout, width=640, height=480):
    """Render each pipeline state to an RGB frame using MuJoCo.

    Args:
        env: Brax environment (must have env.sys.mj_model).
        rollout: List of pipeline_state objects.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of numpy arrays with shape (height, width, 3) dtype uint8.
    """
    mj_model = env.sys.mj_model
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    frames = []
    for pipeline_state in rollout:
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.array(pipeline_state.q)
        mj_data.qvel[:] = np.array(pipeline_state.qd)
        mujoco.mj_forward(mj_model, mj_data)

        renderer.update_scene(mj_data)
        pixels = renderer.render()
        frames.append(pixels)

    renderer.close()
    return frames


def save_video_mp4(frames, output_path, fps=50):
    """Encode RGB frames to an MP4 file using OpenCV.

    Args:
        frames: List of RGB numpy arrays (H, W, 3).
        output_path: Path to write the .mp4 file.
        fps: Frames per second.

    Raises:
        ValueError: If frames list is empty.
        ImportError: If opencv-python-headless is not installed.
    """
    if not frames:
        raise ValueError("No frames to write")

    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python-headless is required for video recording. "
            "Install with: pip install opencv-python-headless"
        )

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def save_diagnostics(diagnostics, output_path, sample_interval=25):
    """Save rollout diagnostics to a human-readable labeled text file.

    Outputs one block per sampled timestep with labeled observation breakdowns
    (yaw_rate, proj_gravity, command, joint_angles, joint_velocities, last_action),
    plus velocity, action, reward, and done.

    Args:
        diagnostics: Dict with 'observations', 'actions', 'rewards', 'dones',
            and 'velocities' arrays.
        output_path: Path to write the .txt file.
        sample_interval: Output every Nth timestep (default 25 = every 0.5s at 50Hz).
    """
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    obs_all = diagnostics['observations']
    act_all = diagnostics['actions']
    rew_all = diagnostics['rewards']
    done_all = diagnostics['dones']
    vel_all = diagnostics['velocities']
    num_steps = obs_all.shape[0]

    def fmt_array(arr):
        return "  ".join(f"{v:12.6f}" for v in arr)

    with open(output_path, 'w') as f:
        f.write(f"# Telemetry (sampled every {sample_interval} steps)\n\n")
        for i in range(0, num_steps, sample_interval):
            obs = obs_all[i]
            f.write(f"--- step {i} ---\n")
            f.write(f"yaw_rate:           {obs[0]:12.6f}\n")
            f.write(f"proj_gravity:       {fmt_array(obs[1:4])}\n")
            f.write(f"command:            {fmt_array(obs[4:7])}\n")
            f.write(f"joint_angles:       {fmt_array(obs[7:16])}\n")
            f.write(f"joint_velocities:   {fmt_array(obs[16:25])}\n")
            f.write(f"last_action:        {fmt_array(obs[25:34])}\n")
            f.write(f"velocity:           {fmt_array(vel_all[i])}\n")
            f.write(f"action:             {fmt_array(act_all[i])}\n")
            f.write(f"reward:             {rew_all[i]:12.6f}\n")
            f.write(f"done:               {done_all[i]:12.6f}\n")
            f.write("\n")


def record_video(env, make_policy, params, output_path,
                 num_steps=250, seed=0, width=640, height=480, fps=50,
                 diagnostics_path=None):
    """High-level entry point: rollout + render + save.

    Args:
        env: Brax environment instance.
        make_policy: Function from ppo.train() that creates an inference fn.
            Called as make_policy(params, deterministic=True).
        params: Trained policy parameters (normalizer, policy, value).
        output_path: Where to save the MP4 file.
        num_steps: Number of simulation steps (default 250 = 5s at 50Hz).
        seed: Random seed for rollout.
        width: Video frame width.
        height: Video frame height.
        fps: Video frames per second.
        diagnostics_path: Where to save the diagnostics text file.
            Defaults to output_path with '_diagnostics.txt' replacing '.mp4'.
    """
    if diagnostics_path is None:
        diagnostics_path = output_path.replace(".mp4", "_diagnostics.txt")

    print("Recording video...")

    inference_fn = make_policy(params, deterministic=True)
    print(f"  Generating rollout ({num_steps} steps)...")
    rollout, diagnostics = generate_rollout(env, inference_fn, num_steps=num_steps, seed=seed)

    print(f"  Rendering {len(rollout)} frames at {width}x{height}...")
    frames = render_frames(env, rollout, width=width, height=height)

    print(f"  Saving video to {output_path}...")
    save_video_mp4(frames, output_path, fps=fps)

    print(f"  Saving diagnostics to {diagnostics_path}...")
    save_diagnostics(diagnostics, diagnostics_path)

    print(f"  Video saved ({len(frames)} frames)")
