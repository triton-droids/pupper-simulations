# pupper-simulations

Official repository for the Triton Pupper Simulations team - a robotics simulation project focused on quadruped locomotion using MuJoCo and Brax.

## Overview

This project provides simulation and reinforcement learning environments for the Bittle quadruped robot. It includes tools for visualization, training locomotion policies, and converting between different model formats (URDF, MJCF).

## Features

- MuJoCo-based physics simulation for quadruped robots
- Brax reinforcement learning environment for locomotion training
- Interactive 3D visualization using MuJoCo viewer
- Model conversion utilities (URDF to MJCF)
- Configurable reward functions for training
- JAX-accelerated training with GPU support

## Installation

### Prerequisites

- Python 3.12 (or compatible version)
- CUDA 12 support (for GPU-accelerated training)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd pupper-simulations
```

2. Create and activate a virtual environment:

```bash
python -m venv simulations-env
source simulations-env/bin/activate  # On Windows: simulations-env\Scripts\activate
```

3. Install basic dependencies:

```bash
pip install -r requirements.txt
```

4. Install training dependencies (optional, for reinforcement learning):

```bash
pip install -r training_requirements.txt
```

## Directory Structure

```
pupper-simulations/
├── assets/                      # Robot models and media files
│   ├── descriptions/           # Robot description files
│   │   └── bittle/            # Bittle quadruped robot
│   │       ├── meshes/        # STL mesh files
│   │       ├── mjcf/          # MuJoCo model files
│   │       ├── urdf/          # URDF robot descriptions
│   │       └── xacro/         # Xacro source files
│   └── media/                 # Media assets
│
├── locomotion/                 # RL training environment
│   ├── bittle_env.py          # Brax environment for Bittle
│   ├── training_helpers.py    # Training utilities
│   ├── training.ipynb         # Training notebook
│   ├── env_test.py            # Environment testing
│   ├── bittle_adapted.xml     # Adapted model configuration
│   └── bittle_adapted_scene.xml  # Scene configuration
│
├── visualization/              # Visualization and simulation tools
│   ├── main.py                # Main visualization script
│   ├── model_converter.py     # URDF to MJCF converter
│   ├── simEnv.py              # Simulation environment
│   ├── constants.py           # Configuration constants
│   └── init.py                # Initialization utilities
│
├── requirements.txt            # Core dependencies
├── training_requirements.txt   # Training-specific dependencies
└── README.md                  # This file
```

## Usage

### Model Visualization

To launch the interactive MuJoCo viewer for inspecting robot models:

```bash
cd visualization
python main.py
```

This will:

1. Convert URDF to MJCF if necessary
2. Load the Bittle robot model
3. Launch the interactive 3D viewer

### Policy Visualization

To visualize trained policies and generate videos:

```bash
python visualize.py
```

This script:

1. Loads a trained policy from ONNX format
2. Runs the policy in the Bittle environment
3. Generates MP4 and GIF videos of the robot's behavior
4. Saves outputs to the `outputs/` directory

**Configuration:** Edit the paths in `visualize.py`:
- `POLICY_PATH` - Path to your ONNX policy file (default: `locomotion/sim-outputs/policies/policy.onnx`)
- `SCENE_PATH` - Path to the scene XML file (default: `locomotion/bittle_adapted_scene.xml`)
- `OUTPUT_DIR` - Output directory for videos (default: `outputs`)

This is useful for:
- Testing policies locally before deploying to the robot
- Iterating on environment design and reward functions
- Creating demonstration videos
- Debugging locomotion behaviors

## Dependencies

### Core Dependencies

- `numpy==2.3.4` - Numerical computing
- `mujoco==3.3.4` - Physics simulation engine
- `xacrodoc==1.3.0` - URDF/Xacro processing

### Training Dependencies

- `jax[cuda12]==0.8.0` - GPU-accelerated numerical computing
- `brax==0.13.0` - Reinforcement learning library
- `ml_collections==1.1.0` - Configuration management
- `mediapy==1.2.4` - Media processing

## Known Issues

### URDF to MJCF Conversion Error

If you encounter this error during URDF to MJCF conversion:

```
model = mujoco.MjModel.from_xml_path(URDF_PATH)
ValueError: Error: error 'inertia must have positive eigenvalues' in alternative for principal axes
Element name 'f_1', id 74
```

**Solution:** Uninstall MuJoCo 3.3.4, install earlier versions (3.2.7, 3.3.1, or 3.3.3), then reinstall MuJoCo 3.3.4 as the primary library:

```bash
pip uninstall mujoco
pip install mujoco==3.2.7  # Or 3.3.1, 3.3.3
pip install mujoco==3.3.4
```

## Contributing

This is the official repository for the Triton Pupper Simulations team. For contributions, please follow the team's development guidelines.

## License

See the repository license file for details.
