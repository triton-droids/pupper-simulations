# Pupper Simulations Setup Guide

Welcome to the Pupper Simulations project! This guide will walk you through setting up your local development environment, connecting to the remote ML training node, and using the training workflow.

## Prerequisites

- Python 3.11 or higher
- Git
- SSH client (built-in on macOS/Linux, Git Bash on Windows)

---

## 1. Initial Setup

### Clone the Repository

```bash
git clone <repo-url>
cd pupper-simulations
```

### Create a New Branch

Always work on your own branch to avoid conflicts with other team members.

```bash
git checkout -b your-name/feature-name
```

**Example:** `git checkout -b oren/improve-locomotion`

### Create Virtual Environment

Set up a Python virtual environment using the project's `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
pip install -e .
```

This installs all required dependencies including MuJoCo, Brax, JAX, and other RL training tools.

### Configure Environment Variables

Create your `.env` file from the example template:

```bash
cp .env.example .env
```

Then edit `.env` with your configuration:

```bash
BRANCH_NAME="your-branch"              # Your git branch name
SSH_KEY_PATH="~/.ssh/id_ed25519"       # Path to your SSH private key
SSH_PASSWORD="your-ssh-password"       # (Optional) SSH password if required
SSH_DIRECTORY="your-directory"         # Your directory on the remote server
SSH_PORT="8000"                        # Port for HTTP server (choose any free port like 8000, 8080, etc.)
DROIDS_IP_ADDRESS="tritondroids@132.249.64.152"  # Remote server address
```

**Important:** The `.env` file is gitignored and contains sensitive information. Never commit it to the repository.

---

## 2. SSH Setup

### Request Access

Before you can use the remote ML node, you need SSH access:

1. Contact **iwebster@ucsd.edu** (Discord: jahovajenkins)
2. Provide your SSH public key for approval
3. Wait for confirmation

### Generate SSH Key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

This creates `~/.ssh/id_ed25519` (private key) and `~/.ssh/id_ed25519.pub` (public key). Share the `.pub` file with iwebster@ucsd.edu.

For detailed SSH setup instructions, see [How to SSH into the ML Node.pdf](./How%20to%20SSH%20into%20the%20ML%20Node.pdf).

### Test Your Connection

```bash
ssh -i ~/.ssh/id_ed25519 tritondroids@132.249.64.152
```

If successful, you'll be logged into the remote server.

### Remote Repository Setup

**CRITICAL:** Only clone YOUR branch on the ML node and work exclusively from it. This prevents interfering with other team members' work.

```bash
# On the remote ML node (after SSH)
git clone -b your-branch-name <repo-url>
cd pupper-simulations

# Create uv environment and install dependencies from pyproject.toml
uv venv
source .venv/bin/activate
uv pip install -e .

cd locomotion
```

---

## 3. Development Workflow

### Remote Training Workflow

#### train.sh - Deploy and Train

The `train.sh` script automates deploying your code to the remote server and starting training.

**What it does:** Commits your changes, pushes to your branch, SSHs into the remote server, pulls the latest code, and runs training.

**Usage:**

```bash
./train.sh              # Full training (10M timesteps, ~30 minutes)
./train.sh --test       # Test run (10K timesteps, ~6 minutes)
```

**Script contents:**

```bash
#!/bin/bash

set -e

# Pull local variables from .env file
source .env

echo "Starting deployment process..."

# 1. Stage, commit, and push current working changes
echo "Staging and comitting changes"
git add .
git commit -m "Deploying latest changes for training"
git push -u origin $BRANCH_NAME

# 2. SSH into the remote server and run training script
echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

ssh -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" << EOF
  set -e
  cd ~/$SSH_DIRECTORY/pupper-simulations/locomotion

  # Pull changes from branch
  git pull

  # Run training script
  uv run train.py $@
EOF
```

#### view-results.sh - View Training Results

The `view-results.sh` script downloads your training results and sets up port forwarding to browse remote files.

**What it does:** Connects to the remote server, downloads the latest training video and policy, opens the video locally, and keeps port forwarding active.

**Usage:**

```bash
./view-results.sh         # View full training results
./view-results.sh --test  # View test run results
```

**Accessing the remote output directory:**

While port forwarding is active, navigate to `http://localhost:$SSH_PORT` in your browser (replace `$SSH_PORT` with your port from `.env`, e.g., `http://localhost:8000`).

This gives you access to the remote `locomotion/` directory where you can:

- Browse all training outputs
- View TensorBoard logs
- Download additional checkpoints
- Inspect training metrics

**To stop port forwarding:**

```bash
pkill -f 'ssh.*8000:localhost:8000'  # Replace 8000 with your SSH_PORT
```

#### Typical Iteration Process

1. **Make changes** to your code locally (e.g., modify `bittle_env.py`, `training_config.py`)
2. **Test locally** (optional but recommended): `python locomotion/train.py --test`
3. **Deploy and train**: `./train.sh` or `./train.sh --test`
4. **Wait** for training to complete (~6 min for test, ~30 min for full)
5. **View results**: `./view-results.sh` or `./view-results.sh --test`
6. **Analyze** the training video and metrics at `http://localhost:$SSH_PORT`
7. **Iterate** based on performance, repeat from step 1

---

## 4. Local Visualization

For inspecting robot models and MJCF/URDF files before training, use the visualization tool:

```bash
python asset-visualization/main.py
```

This opens an interactive 3D viewer where you can load and inspect the Bittle robot model. Useful for debugging model configurations and understanding joint movements.

---

## 5. Opening a Pull Request

Before opening a pull request:

1. **Test your changes** - Run both local and remote tests to ensure everything works
2. **Check the build** - Make sure `pip install -e .` succeeds with no errors
3. **Review your changes** - Use `git diff` to verify you're only committing intended changes
4. **Update documentation** if you've added new features or changed workflows

**To open a PR:**

```bash
# Push your branch if you haven't already
git push -u origin your-name/feature-name

# Use GitHub's web interface or gh CLI
gh pr create --base main --head your-name/feature-name
```

In your PR description:

- Summarize what changed and why
- Include training results (videos, metrics) if applicable
- Mention any breaking changes or dependencies

The team will review your code before merging to `main`.

---

## 6. Additional Resources

- **Main README**: [README.md](../README.md) - Project overview and architecture
- **SSH Guide**: [How to SSH into the ML Node.pdf](./How%20to%20SSH%20into%20the%20ML%20Node.pdf) - Detailed SSH setup
- **Need help?** Contact iwebster@ucsd.edu (Discord: jahovajenkins)

---

**Happy training! 🤖🐕**
