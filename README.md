# pupper-simulations

Simulation and reinforcement learning for the Bittle quadruped robot, using MuJoCo and Brax.

## Overview

This project trains locomotion policies for the Bittle robot via RL on a remote GPU node, then visualizes the results locally with an interactive MuJoCo teleop viewer.

## Installation

Requires Python 3.11+. All dependencies are declared in `pyproject.toml`.

```bash
git clone <repository-url>
cd pupper-simulations
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the interactive viewer, you also need `mjpython` (ships with the `mujoco` package).

## Directory Structure

```
pupper-simulations/
├── locomotion/                    # RL training environment + teleop
│   ├── constants.py               # Shared robot parameters (single source of truth)
│   ├── bittle_env.py              # Brax environment for Bittle
│   ├── teleop.py                  # Interactive MuJoCo teleop viewer
│   ├── train.py                   # Training entry point
│   ├── training_config.py         # Hyperparameters
│   ├── onnx_export.py             # Export trained policy to ONNX
│   ├── domain_randomization.py    # Domain randomization (not yet integrated)
│   ├── env_test.py                # Quick env build test
│   ├── bittle_adapted.xml         # MuJoCo robot model
│   ├── bittle_adapted_scene.xml   # Scene with floor, lighting, keyframe
│   └── outputs/                   # Training outputs (gitignored)
│
├── assets/                        # Robot meshes and description files
│   └── descriptions/bittle/       # STL meshes, URDF, MJCF, Xacro
│
├── tests/
│   ├── test_visualize.sh          # Shell tests for visualize.sh
│   └── test_constants.py          # Constants consistency tests
│
├── docs/
│   └── SETUP.md                   # Detailed setup and SSH guide
│
├── visualize.sh                   # Download policy + launch teleop
├── pyproject.toml                 # Python project and dependencies
└── .env.example                   # Template for SSH credentials
```

## Workflow

### 1. Train (remote SSH node)

```bash
ssh -i ~/.ssh/your-key tritondroids@132.249.64.152
cd pupper-simulations/locomotion
python train.py              # full training
python train.py --test       # quick test run
```

Training outputs a policy to `locomotion/outputs/policy.onnx`.

### 2. Download + Visualize (local)

```bash
./visualize.sh               # download policy from remote + launch teleop
./visualize.sh --dry-run     # print commands without executing
./visualize.sh --download-only  # download only, skip teleop
./visualize.sh --video       # also download training video
```

### 3. Interactive Teleop Controls

| Key | Action |
|-----|--------|
| W / S | Forward / backward velocity |
| A / D | Left / right velocity |
| Left / Right arrows | Yaw |
| Space | Zero all commands |
| R | Reset simulation |
| Q | Quit |

### Model Inspection

To view the robot model without a policy:

```bash
mjpython locomotion/teleop.py --no-policy --xml-path locomotion/bittle_adapted_scene.xml
```

Or use the built-in MuJoCo viewer:

```bash
python -m mujoco.viewer --mjcf locomotion/bittle_adapted_scene.xml
```

## Configuration

Environment variables (in `.env`, see `.env.example`):

| Variable | Description |
|----------|-------------|
| `SSH_KEY_PATH` | Path to your SSH private key |
| `SSH_DIRECTORY` | Your project directory on the remote server |
| `DROIDS_IP_ADDRESS` | Remote server address (user@host) |

## Dependencies

All dependencies are managed in `pyproject.toml`. Key packages:

- **mujoco** — physics simulation
- **brax** — RL environment and training
- **jax[cuda12]** — GPU-accelerated training
- **onnxruntime** — policy inference for teleop

See [docs/SETUP.md](docs/SETUP.md) for detailed setup instructions.
