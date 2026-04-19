# Pupper New

MuJoCo + Brax training, sweeping, and visualization for the Bittle quadruped.

This repo is organized around three questions:

- What can the robot be asked to do?
- How is training run and monitored?
- What scripts do you launch directly?

## Repository Layout

```text
Pupper New/
|-- README.md
|-- pyproject.toml
|-- Scripts/
|   |-- train.py
|   |-- sweep.sh
|   |-- visualize.py
|   `-- Outputs/
|-- locomotion/
|   |-- __init__.py
|   |-- paths.py
|   |-- domain_randomization.py
|   |-- onnx_export.py
|   |-- train_compat.py
|   |-- training/
|   |   |-- __init__.py
|   |   |-- run.py
|   |   |-- config.py
|   |   |-- helpers.py
|   |   `-- monitor.py
|   |-- tasks/
|   |   |-- __init__.py
|   |   |-- README.md
|   |   |-- bittle_walk_env.py
|   |   |-- bittle_walking_hparams.json
|   |   |-- bittle_dance_env.py
|   |   `-- bittle_dance_hparams.json
|   |-- sweeps/
|   |   |-- README.md
|   |   |-- hparam_sweep.py
|   |   `-- training_budget_and_batching_sweep.json
|   `-- export/
|       `-- convert_onnx_ir_version.py
|-- asset_visualization/
|-- assets/
|-- docs/
`-- tests/
```

## What Lives Where

### `Scripts/`

This is the user-facing layer.

- `Scripts/train.py` is the main local training entrypoint.
- `Scripts/sweep.sh` launches a remote hyperparameter sweep and syncs results back.
- `Scripts/visualize.py` loads a policy and renders a rollout.
- `Scripts/Outputs/` is the default output root for training runs, sweep runs, and visualization output.

### `locomotion/training/`

This is the training implementation layer.

- `run.py` wires the whole training job together.
- `config.py` holds task-aware PPO presets.
- `helpers.py` handles CLI parsing, logging, and checkpoint callbacks.
- `monitor.py` writes final metrics, plots, and videos.

`locomotion/train_compat.py` is only a compatibility shim for older commands. The preferred entrypoint is `Scripts/train.py`.

### `locomotion/tasks/`

This is the task-definition layer.

- `bittle_walk_env.py` defines the walking task.
- `bittle_dance_env.py` defines the dance task.
- `bittle_walking_hparams.json` and `bittle_dance_hparams.json` hold task-side hyperparameter sweep entries.
- `locomotion/tasks/README.md` explains what those task-side hyperparameters do in plain language.

### `locomotion/sweeps/`

This is the sweep-coordination layer.

- `README.md` explains the trainer-side sweep parameters in plain language.
- `training_budget_and_batching_sweep.json` controls trainer-side values like total training budget, batch size, minibatching, and parallel environment count.
- `hparam_sweep.py` combines the trainer-side JSON with the task-side JSON for the selected task and runs every combination.

### `assets/` and `asset_visualization/`

- `assets/` contains the Bittle MJCF, meshes, URDF, and related robot-description files.
- `asset_visualization/` contains model-inspection utilities for opening and checking the robot in MuJoCo.

## Installation

### Prerequisites

- Python 3.11 or newer
- Git
- CUDA 12 compatible environment if you plan to train on GPU-backed JAX
- Git Bash on Windows if you want to run `Scripts/sweep.sh`

### Setup

```bash
git clone <repo-url>
cd "Pupper New"
python -m venv .venv
```

Activate the virtual environment:

```bash
# PowerShell
.venv\Scripts\Activate.ps1

# Git Bash
source .venv/Scripts/activate
```

Install the project:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[training]
pip install -e .[dev]
```

## Main Workflow

### 1. Train locally

Quick smoke test:

```bash
python Scripts/train.py --test
```

Full walking run:

```bash
python Scripts/train.py
```

Full dance run:

```bash
python Scripts/train.py --task dance
```

By default, training outputs land in `Scripts/Outputs/` under a task-specific folder such as:

- `Scripts/Outputs/bittle_walking_train_latest`
- `Scripts/Outputs/bittle_dance_train_latest`
- `Scripts/Outputs/bittle_walking_test_latest`
- `Scripts/Outputs/bittle_dance_test_latest`

Each run folder contains logs, checkpoints, model exports, and final human-facing artifacts such as:

- `metrics/final_metrics.json`
- `plots/final_progress.png`
- `videos/final_video.mp4`
- `training_summary.json`

### 2. Run a sweep

```bash
bash Scripts/sweep.sh
```

Current sweep flow:

1. `Scripts/sweep.sh` loads `.env`, prepares the remote command, and reserves the next local folder name.
2. The sweep runner reads the outer-loop list from `locomotion/sweeps/training_budget_and_batching_sweep.json`.
3. It reads the inner-loop list from the matching task JSON in `locomotion/tasks/`.
4. For each one training-budget entry, it loops through every task-hyperparameter entry.
5. Results sync back into a numbered folder under `Scripts/Outputs/`.

In plain terms:

- first pick one trainer-side case
- then try all task-side cases inside that one trainer-side case
- then move to the next trainer-side case

Local sweep outputs now look like this:

```text
Scripts/Outputs/
`-- Sweep #N/
    |-- results.jsonl
    |-- leaderboard.json
    |-- best_trial.json
    |-- trial_001/
    |-- trial_002/
    `-- ...
```

Each `trial_...` folder is one complete combined run. It contains:

- `parameters.txt`
- `training_overrides.json`
- `task_overrides.json`
- `combined_overrides.json`
- `trial_result.json`
- `training_summary.json`
- `metrics/final_metrics.json`
- `plots/final_progress.png`
- `videos/final_video.mp4`
- model exports and checkpoints

Important distinction:

- `locomotion/sweeps/training_budget_and_batching_sweep.json` changes how PPO training is budgeted and sliced up.
- `locomotion/tasks/*.json` changes task-side environment behavior and reward weighting.

### 3. Visualize a trained policy

```bash
python Scripts/visualize.py
```

You can also pass an explicit ONNX file:

```bash
python Scripts/visualize.py Scripts/Outputs/bittle_dance_train_latest/policy.onnx
```

### 4. Inspect the robot model

```bash
python asset_visualization/main.py
```

### 5. Run the test suite

```bash
python -m unittest
```

Or run a smaller targeted subset:

```bash
python -m unittest tests.test_env_build tests.test_hparam_sweep
```

## Tasks

There are two canonical trainable tasks:

- `walking`
- `dance`

`locomotion` is still accepted as a compatibility alias for `walking`.

The task-specific hyperparameter reference lives in [locomotion/tasks/README.md](locomotion/tasks/README.md).

## Outputs

`Scripts/Outputs/` is the default output root across the current workflow.

That includes:

- local train/test runs
- visualization output
- numbered sweep folders like `Sweep #0`, `Sweep #1`, and so on

Relative output paths are resolved under `Scripts/Outputs/` by `locomotion/paths.py`, so ad hoc names like `run_01` do not end up in whatever directory you happened to launch from.

## Notes

- `Scripts/sweep.sh` currently defaults to the dance task when it launches the remote sweep command.
- The direct Python sweep runner in `locomotion/sweeps/hparam_sweep.py` supports task selection and task-side JSON overrides directly.
- `locomotion/train_compat.py` exists only to avoid breaking older commands while the repo transitions to the `Scripts/train.py` entrypoint.

## Additional Documentation

- `docs/SETUP.md` for environment and ML-node setup
- `docs/How to SSH into the ML Node.pdf` for SSH access instructions
