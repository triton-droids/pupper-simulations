# Pupper New

Reorganizes repository for the Triton Pupper Simulations team. This project centers on MuJoCo and Brax-based simulation, reinforcement learning, and visualization for the Bittle quadruped.

## Overview

This repository contains:

- a Brax locomotion environment for Bittle
- training utilities and hyperparameter sweep tooling
- visualization scripts for trained policies
- asset conversion and inspection tools for the robot model
- MuJoCo assets, meshes, MJCF files, URDF, and Xacro sources

The current repository is organized around responsibilities:

- **root files** define the project and environment
- **`Scripts/`** contains user-facing entry scripts
- **`locomotion/`** contains the training and environment logic
- **`asset_visualization/`** contains model conversion and model inspection utilities
- **`assets/`** contains robot description files and meshes
- **`tests/`** contains validation helpers
- **`docs/`** contains setup and infrastructure documentation

## Current Repository Structure

```text
pupper-simulations/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env                         # local only, gitignored
│
├── Scripts/
│   ├── sweep.sh                 # deploy and run remote hyperparameter sweeps
│   └── visualize.py             # load ONNX policy and render rollout video
│
├── locomotion/
│   ├── bittle_env.py            # Brax environment definition for Bittle
│   ├── domain_randomization.py  # domain randomization logic for training
│   ├── onnx_export.py           # export trained policy to ONNX
│   ├── train.py                 # main training entrypoint
│   ├── training_config.py       # training configuration container
│   ├── training_helpers.py      # logging, checkpointing, CLI helpers
│   ├── training_monitor.py      # metrics, plots, and training monitoring
│   ├── export/
│   │   └── convert_onnx_ir_version.py
│   └── sweeps/
│       ├── hparam_sweep.py      # sweep runner
│       └── trials_2080ti_screen.json
│
├── asset_visualization/
│   ├── constants.py             # model and path constants
│   ├── logging_utils.py         # logger setup
│   ├── main.py                  # interactive MuJoCo model viewer
│   └── model_converter.py       # URDF to MJCF conversion utilities
│
├── assets/
│   └── descriptions/
│       └── bittle/
│           ├── meshes/
│           │   └── stl/
│           ├── mjcf/
│           │   ├── bittle.xml
│           │   ├── bittle_adapted.xml
│           │   ├── bittle_adapted_scene.xml
│           │   ├── bittle_assets.xml
│           │   └── bittle_body.xml
│           ├── urdf/
│           │   └── bittle.urdf
│           └── xacro/
│
├── tests/
│   └── test_env_build.py        # quick environment build check
│
├── docs/
│   ├── SETUP.md
│   └── How to SSH into the ML Node.pdf
│
└── pupper_simulations.egg-info/ # generated packaging metadata
```

## Installation

### Prerequisites

- Python 3.11 or newer
- Git
- CUDA 12 compatible environment if you plan to use GPU-backed JAX training
- Git Bash on Windows if you want to run the shell scripts in `Scripts/`

### Setup

1. Clone the repository:

```bash
git clone <repo-url>
cd pupper-simulations
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows (PowerShell): .venv\Scripts\Activate.ps1
# On Windows (Git Bash): source .venv/Scripts/activate
```

3. Install the project from `pyproject.toml`:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[training]
pip install -e .[dev]
```

## What Each Area Does

### `Scripts/`
User-facing operational entrypoints.

- `sweep.sh` pushes your branch, connects to the remote training node, runs a hyperparameter sweep, and syncs artifacts back locally.
- `visualize.py` loads a trained ONNX policy, runs rollout in the Bittle environment, and writes MP4 and GIF outputs.

### `locomotion/`
Core training and environment code.

- defines the Bittle Brax environment
- defines training configuration and monitoring
- runs training
- exports trained policies
- runs parameter sweeps

### `asset_visualization/`
Model-side utilities.

- converts model descriptions into MJCF components when needed
- opens the MuJoCo viewer to inspect the Bittle model
- provides asset-related constants and logging helpers

### `assets/`
Robot description and simulation resources.

- MJCF scene and body files
- URDF source
- Xacro source
- STL meshes used by the MuJoCo model

### `tests/`
Quick validation helpers.

- `test_env_build.py` is a smoke test that checks whether the Bittle environment can be constructed successfully.

## Usage

### 1. Train a policy locally

From the repository root:

```bash
python locomotion/train.py --test
```

For a full run:

```bash
python locomotion/train.py
```

The training code uses the Bittle scene XML under:

```text
assets/descriptions/bittle/mjcf/bittle_adapted_scene.xml
```

### 2. Run a hyperparameter sweep on the remote node

From the repository root or by launching the script directly in PyCharm:

```bash
bash Scripts/sweep.sh
```

Current behavior of `Scripts/sweep.sh`:

- loads environment variables from the repository root `.env`
- commits and pushes the current branch
- SSHes into the remote ML node
- runs `locomotion/sweeps/hparam_sweep.py`
- syncs sweep artifacts back to a local folder under:

```text
Scripts/outputs/sweeps/
```

If you want to change remote connection details, branch name, or key paths, edit `.env` and consult `docs/SETUP.md`.

### 3. Visualize a trained policy

```bash
python Scripts/visualize.py
```

Or provide an explicit ONNX policy path:

```bash
python Scripts/visualize.py outputs/bittle_train_latest/policy.onnx
```

`Scripts/visualize.py` searches common policy locations and writes rendered outputs to:

```text
outputs/visualize/
```

### 4. Inspect the robot model in MuJoCo

```bash
python asset_visualization/main.py
```

This launches the MuJoCo viewer for the Bittle model and uses the asset conversion utilities if the generated MJCF components are missing.

### 5. Smoke test the environment build

```bash
python tests/test_env_build.py
```

This is useful after reorganizing asset paths or changing MJCF files.

## Dependencies

Project dependencies are defined in `pyproject.toml`, not in `requirements.txt` files.

Core dependencies currently include:

- `numpy`
- `mujoco`
- `xacrodoc`
- `ml_collections`
- `jax[cuda12]`
- `brax`
- `opencv-python`
- `matplotlib`
- `onnx`
- `onnxruntime`

Optional groups:

- `training`: plotting, Pillow, TensorBoardX
- `dev`: Jupyter and notebook tooling

## Development Notes

- Work on your own branch.
- Use `docs/SETUP.md` for ML node and SSH setup.
- Keep `.env`, `.venv`, local outputs, and IDE files out of version control.
- If you move MJCF or mesh assets, update both Python paths and XML-relative asset references.

## Known Caveats

- The repository still contains some generated or local-only material in exports and archives that should not be treated as source of truth.
- Local sweep artifacts currently sync into `Scripts/outputs/`, while visualization writes to `outputs/visualize/`. That is the current behavior, even though a future cleanup may consolidate outputs into a single top-level `outputs/` directory.
- Asset path changes can break MuJoCo loading if `meshdir` or included MJCF asset references are not updated consistently.

## Additional Documentation

- `docs/SETUP.md` for environment and ML node setup
- `docs/How to SSH into the ML Node.pdf` for SSH access instructions

## Contributing

For contributions, follow the team workflow documented in `docs/SETUP.md`, keep changes scoped to your branch, and include enough context for others to reproduce training or visualization results.
