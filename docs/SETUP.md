# Pupper Simulations Setup Guide

Welcome to the Pupper Simulations project! This guide will walk you through setting up your local development environment, connecting to the remote ML training node, and using the training workflow.

## Prerequisites

- Python 3.11 or higher
- Git
- SSH client (built-in on macOS/Linux, Git Bash or OpenSSH on Windows)
- CUDA 12 support (optional, for local GPU training)

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

**macOS/Linux:**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Windows (Command Prompt):**
```cmd
copy .env.example .env
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

**macOS/Linux:**
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

**Windows (PowerShell or Command Prompt):**
```cmd
ssh-keygen -t ed25519 -C "your-email@example.com"
```

This creates `~/.ssh/id_ed25519` (private key) and `~/.ssh/id_ed25519.pub` (public key) on macOS/Linux, or `C:\Users\YourName\.ssh\id_ed25519` on Windows. Share the `.pub` file with iwebster@ucsd.edu.

**Windows users:** Make sure OpenSSH is installed. On Windows 10/11, you can install it via Settings → Apps → Optional Features → Add a feature → OpenSSH Client.

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

**macOS/Linux:**
```bash
pkill -f 'ssh.*8000:localhost:8000'  # Replace 8000 with your SSH_PORT
```

**Windows (PowerShell):**
```powershell
Get-Process | Where-Object {$_.ProcessName -eq "ssh"} | Stop-Process
```

#### Typical Iteration Process

1. **Make changes** to your code locally (e.g., modify `bittle_env.py`, `training_config.py`)
2. **Test locally** (optional but recommended): `python locomotion/train.py --test`
3. **Deploy and train**: `./train.sh` or `./train.sh --test`
4. **Wait** for training to complete (~6 min for test, ~30 min for full)
5. **View results**: `./view-results.sh` or `./view-results.sh --test`
6. **Visualize locally**: `python visualize.py` to see detailed policy behavior
7. **Analyze** the training video, local visualization, and metrics at `http://localhost:$SSH_PORT`
8. **Iterate** based on performance, repeat from step 1

**Note for Windows users:** The shell scripts (`train.sh`, `view-results.sh`) require Bash. Use Git Bash, WSL (Windows Subsystem for Linux), or manually run the commands in the scripts using PowerShell equivalents.

---

## 4. Local Testing and Visualization

### Model Visualization

For inspecting robot models and MJCF/URDF files before training, use the asset visualization tool:

```bash
python asset-visualization/main.py
```

This opens an interactive 3D viewer where you can load and inspect the Bittle robot model. Useful for debugging model configurations and understanding joint movements.

### Policy Visualization and Environment Iteration

To test trained policies locally and iterate on the environment:

```bash
python visualize.py
```

**What it does:**
- Loads a trained policy from ONNX format
- Runs the policy in the Bittle simulation environment
- Generates MP4 and GIF videos showing the robot's behavior
- Saves outputs to the `outputs/` directory

**Configuration:** Edit `visualize.py` to customize:
- `POLICY_PATH` - Path to your ONNX policy file
- `SCENE_PATH` - Scene configuration file
- `DURATION` / `NUM_STEPS` - Length of visualization
- `RENDER_WIDTH` / `RENDER_HEIGHT` - Video resolution

**Typical workflow for environment iteration:**

1. **Make changes** to reward functions, observations, or environment parameters in `locomotion/bittle_env.py`
2. **Train remotely** using `./train.sh --test` (quick 6-minute test run)
3. **Download results** using `./view-results.sh --test`
4. **Visualize locally** with `python visualize.py` to see the policy behavior
5. **Analyze** the videos and metrics to understand what needs improvement
6. **Iterate** - repeat from step 1 with refined changes

This allows rapid iteration without waiting for full training runs.

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
