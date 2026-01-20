import jax
import jax.numpy as jp
import numpy as np
import cv2
from bittle_env import BittleEnv 

def record_moving_pose_opencv(xml_path: str, output_path: str = 'bittle_slight_move.mp4'):
    # 1. Initialize environment
    env = BittleEnv(xml_path=xml_path)
    
    # 2. Setup JIT functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # 3. Initialize state
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    
    fps = int(1 / env.dt)
    seconds = 5
    num_steps = seconds * fps
    
    rollout = []
    print(f"Simulating {seconds} seconds with micro-movements...")
    
    # 4. Simulation Loop
    for i in range(num_steps):
        rollout.append(state.pipeline_state)
        
        # Create a small oscillation (amplitude 0.1 out of 1.0 max action)
        # Time-based frequency: 0.5 Hz (one full cycle every 2 seconds)
        t = i * env.dt
        oscillation = 0.1 * jp.sin(2 * jp.pi * 0.5 * t)
        
        # Apply to all joints (9 actuators)
        # Positive/Negative values shift joints slightly away from default pose
        action = jp.ones(env.sys.nu) * oscillation
        
        state = jit_step(state, action)
    
    # 5. Render
    print("Rendering frames...")
    frames = env.render(rollout, width=640, height=480)
    
    # 6. Save with OpenCV
    print(f"Writing video to {output_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
        
    out.release()
    print("Done!")

if __name__ == "__main__":
    record_moving_pose_opencv("bittle_adapted_scene.xml")