# Pupper Simulations Setup Guide

Welcome to the Pupper Simulations project! This guide walks you through setting up your local development environment, connecting to the remote ML training node, and using the training workflow.

## Prerequisites

- Python 3.11 or higher
- Git
- SSH client (built-in on macOS/Linux, Git Bash or OpenSSH on Windows)
- CUDA 12 support (on the remote training node)

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

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .
```

This installs all required dependencies including MuJoCo, Brax, JAX, and other RL training tools.

**Note for Windows users:** If you encounter execution policy errors when activating the virtual environment, run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Configure Environment Variables

Create your `.env` file from the example template:

```bash
cp .env.example .env
```

Then edit `.env` with your configuration:

```bash
SSH_KEY_PATH="~/.ssh/id_ed25519"       # Path to your SSH private key
SSH_DIRECTORY="your-directory"         # Your project directory on the remote server
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

### Training (Remote)

SSH into the ML node and run training:

```bash
ssh -i ~/.ssh/your-key tritondroids@132.249.64.152
cd pupper-simulations/locomotion
python train.py              # Full training (~30 minutes)
python train.py --test       # Quick test run (~6 minutes)
```

### Download + Visualize (Local)

Use `visualize.sh` to download the trained policy and launch the interactive teleop viewer:

```bash
./visualize.sh               # Download policy + launch teleop
./visualize.sh --dry-run     # Print commands without executing
./visualize.sh --download-only  # Download only, no teleop
./visualize.sh --video       # Also download training video
```

Or run the teleop directly with an already-downloaded policy:

```bash
mjpython locomotion/teleop.py --policy locomotion/outputs/policy.onnx
```

### Model Inspection

To inspect the robot model without a trained policy:

```bash
mjpython locomotion/teleop.py --no-policy --xml-path locomotion/bittle_adapted_scene.xml
```

Or use the built-in MuJoCo viewer:

```bash
python -m mujoco.viewer --mjcf locomotion/bittle_adapted_scene.xml
```

### Typical Iteration Process

1. **Make changes** to your code locally (e.g., modify `locomotion/bittle_env.py`, `locomotion/training_config.py`)
2. **Push changes** to your branch
3. **SSH into ML node**, pull changes, and train: `cd locomotion && python train.py --test`
4. **Download + visualize**: `./visualize.sh`
5. **Iterate** based on performance, repeat from step 1

**Note for Windows users:** The shell scripts (`visualize.sh`) require Bash. Use Git Bash or WSL (Windows Subsystem for Linux).

---

## 4. Opening a Pull Request

Before opening a pull request:

1. **Test your changes** — Run tests to ensure everything works
2. **Check the build** — Make sure `pip install -e .` succeeds with no errors
3. **Review your changes** — Use `git diff` to verify you're only committing intended changes
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

## 5. Additional Resources

- **Main README**: [README.md](../README.md) — Project overview and workflow
- **SSH Guide**: [How to SSH into the ML Node.pdf](./How%20to%20SSH%20into%20the%20ML%20Node.pdf) — Detailed SSH setup
- **Need help?** Contact iwebster@ucsd.edu (Discord: jahovajenkins)
