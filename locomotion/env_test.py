"""
Comprehensive sanity-check script for BittleEnv.

Run from the locomotion/ directory:
    python env_test.py

Runs 4 checks with real physics and prints PASS/FAIL for each.
"""

import jax
import jax.numpy as jp
import numpy as np
from brax import envs, math

from bittle_env import BittleEnv
from video_recorder import render_frames, save_video_mp4

# ── Setup ───────────────────────────────────────────────────────────────
envs.register_environment("bittle", BittleEnv)

xml_path = "bittle_adapted_scene.xml"
env = envs.get_environment("bittle", xml_path=xml_path, enable_kicks=False)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Warm-up JIT (first call triggers compilation)
_warmup_state = jit_reset(jax.random.PRNGKey(99))
_warmup_state = jit_step(_warmup_state, jp.zeros(env.sys.nu))

results = []  # list of (name, passed)

# ── Check 1: Environment Build ─────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 1: Environment Build")
print("=" * 60)

passed = True
details = []

nu = env.sys.nu
nq = env.sys.nq
nv = env.sys.nv
n_legs = len(env._lower_leg_body_id)

details.append(f"  nu={nu} (expected 9)")
details.append(f"  nq={nq} (expected 16)")
details.append(f"  nv={nv} (expected 15)")
details.append(f"  lower leg bodies found: {n_legs} (expected 4)")

if nu != 9:
    passed = False
if nq != 16:
    passed = False
if nv != 15:
    passed = False
if n_legs != 4:
    passed = False

for d in details:
    print(d)
print(f"{'PASS' if passed else 'FAIL'}: Environment Build")
results.append(("Environment Build", passed))

# ── Check 2: Standing Stability ─────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 2: Standing Stability (zero action, 100 steps = 2s)")
print("=" * 60)

state = jit_reset(jax.random.PRNGKey(0))
zero_action = jp.zeros(env.sys.nu)

passed = True
for i in range(100):
    state = jit_step(state, zero_action)

# Read final values
done_val = float(state.done)
base_height = float(state.pipeline_state.x.pos[env._base_body_id, 2])
up_vec = math.rotate(
    jp.array([0.0, 0.0, 1.0]),
    state.pipeline_state.x.rot[env._base_body_id],
)
up_z = float(up_vec[2])

print(f"  final base height: {base_height:.4f} m")
print(f"  up-vector z:       {up_z:.4f}")
print(f"  done flag:         {done_val}")

if done_val != 0.0:
    passed = False
    print("  FAIL reason: terminated early")
if not (0.03 <= base_height <= 0.15):
    passed = False
    print(f"  FAIL reason: base height {base_height:.4f} outside [0.03, 0.15]")
if up_z <= 0.7:
    passed = False
    print(f"  FAIL reason: up-vector z {up_z:.4f} <= 0.7")

print(f"{'PASS' if passed else 'FAIL'}: Standing Stability")
results.append(("Standing Stability", passed))

# ── Check 3: Sinusoidal Action Rollout (~3s) ────────────────────────────
print("\n" + "=" * 60)
print("CHECK 3: Sinusoidal Action Rollout (150 steps = 3s)")
print("=" * 60)

state = jit_reset(jax.random.PRNGKey(1))
rollout = [state.pipeline_state]
rewards = []
num_steps = 150
frequencies = np.linspace(0.5, 2.0, env.sys.nu)  # Hz per joint
amplitude = 0.3
dt = 0.02  # control dt

passed = True
for i in range(num_steps):
    t = i * dt
    action = jp.array(amplitude * np.sin(2.0 * np.pi * frequencies * t))
    state = jit_step(state, action)
    rollout.append(state.pipeline_state)
    rewards.append(float(state.reward))

done_val = float(state.done)
base_height = float(state.pipeline_state.x.pos[env._base_body_id, 2])
all_finite = all(np.isfinite(r) for r in rewards)

print(f"  final base height: {base_height:.4f} m")
print(f"  done flag:         {done_val}")
print(f"  all rewards finite: {all_finite}")

if done_val != 0.0:
    passed = False
    print("  FAIL reason: terminated during rollout")
if not all_finite:
    passed = False
    print("  FAIL reason: non-finite reward detected")

# Video rendering (best-effort)
video_saved = False
try:
    frames = render_frames(env, rollout, width=640, height=480)
    save_video_mp4(frames, "env_test_sinusoidal.mp4", fps=50)
    video_saved = True
    print("  video saved: env_test_sinusoidal.mp4")
except Exception as e:
    print(f"  video skipped: {e}")

print(f"{'PASS' if passed else 'FAIL'}: Sinusoidal Action Rollout")
results.append(("Sinusoidal Action Rollout", passed))

# ── Check 4: Reset Consistency ──────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 4: Reset Consistency (same seed → identical state)")
print("=" * 60)

seed_key = jax.random.PRNGKey(42)
state_a = jit_reset(seed_key)
state_b = jit_reset(seed_key)

qpos_match = jp.allclose(state_a.pipeline_state.q, state_b.pipeline_state.q)
qvel_match = jp.allclose(state_a.pipeline_state.qd, state_b.pipeline_state.qd)
reward_match = jp.allclose(state_a.reward, state_b.reward)
done_match = jp.allclose(state_a.done, state_b.done)

passed = bool(qpos_match and qvel_match and reward_match and done_match)

print(f"  qpos match:   {bool(qpos_match)}")
print(f"  qvel match:   {bool(qvel_match)}")
print(f"  reward match: {bool(reward_match)}")
print(f"  done match:   {bool(done_match)}")
print(f"{'PASS' if passed else 'FAIL'}: Reset Consistency")
results.append(("Reset Consistency", passed))

# ── Check 5: Flipped Termination ────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 5: Flipped Termination (robot on its back → done=1)")
print("=" * 60)

# Build a qpos with the robot flipped 180° around the x-axis.
# Normal base quat is [1,0,0,0] (upright). Flipped: [0,1,0,0].
from locomotion.constants import DEFAULT_POSE as _DEFAULT_POSE, INIT_QPOS_BASE as _INIT_BASE

flipped_qpos = jp.zeros(env.sys.nq)
flipped_base = jp.array(_INIT_BASE).at[3:7].set(jp.array([0.0, 1.0, 0.0, 0.0]))
# Raise z so the robot starts above ground before falling
flipped_base = flipped_base.at[2].set(0.15)
flipped_qpos = flipped_qpos.at[:7].set(flipped_base)
flipped_qpos = flipped_qpos.at[env._q_joint_start:].set(jp.array(_DEFAULT_POSE))

flipped_pipeline = env.pipeline_init(flipped_qpos, jp.zeros(env.sys.nv))

# Graft the flipped pipeline_state into a normal reset state
state = jit_reset(jax.random.PRNGKey(7))
state = state.replace(pipeline_state=flipped_pipeline)

# Step once and check termination
state = jit_step(state, jp.zeros(env.sys.nu))

done_val = float(state.done)
up_vec = math.rotate(
    jp.array([0.0, 0.0, 1.0]),
    state.pipeline_state.x.rot[env._base_body_id],
)
up_z = float(up_vec[2])

print(f"  up-vector z: {up_z:.4f} (expected < 0.5)")
print(f"  done flag:   {done_val} (expected 1.0)")

passed = done_val == 1.0
if not passed:
    print("  FAIL reason: flipped robot did not terminate")

print(f"{'PASS' if passed else 'FAIL'}: Flipped Termination")
results.append(("Flipped Termination", passed))

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n_passed = sum(1 for _, p in results if p)
n_total = len(results)
print(f"SUMMARY: {n_passed}/{n_total} PASSED", end="")
failed = [name for name, p in results if not p]
if failed:
    print(f", {len(failed)} FAILED: {', '.join(failed)}")
else:
    print()
print("=" * 60)
