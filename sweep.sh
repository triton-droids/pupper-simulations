#!/usr/bin/env bash
set -euo pipefail

# Load env vars from .env next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/.env"
set +a

echo "Starting deployment process for SWEEP..."
echo "Staging and committing changes"

git add .
git commit -m "Deploying latest changes for sweep" || echo "No changes to commit"
git push -u origin "$BRANCH_NAME"

# Ensure local ssh-agent has GitHub key loaded (needed for -A forwarding)
if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
fi
ssh-add "$GITHUB_KEY_PATH" >/dev/null 2>&1 || true

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

# Allow overriding trials file path from your local shell if desired
SWEEP_TRIALS_JSON="${SWEEP_TRIALS_JSON:-sweeps/trials_2080ti_screen.json}"

ssh -A -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" bash -s -- \
  "$BRANCH_NAME" "$SSH_DIRECTORY" "$GITHUB_REPO_SSH" "$SWEEP_TRIALS_JSON" "$@" <<'REMOTE'
set -euo pipefail

BRANCH_NAME="$1"; shift
SSH_DIRECTORY="$1"; shift
GITHUB_REPO_SSH="$1"; shift
SWEEP_TRIALS_JSON="$1"; shift

# Pick GPU and tame JAX memory behavior (adjust as needed)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

mkdir -p "$SSH_DIRECTORY"
cd "$SSH_DIRECTORY"

if [ ! -d .git ]; then
  git clone "$GITHUB_REPO_SSH" .
fi

git fetch origin
git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"
git reset --hard "origin/$BRANCH_NAME"

cd locomotion

echo "Running hyperparameter sweep..."
echo "Trials: $SWEEP_TRIALS_JSON"

# Run sweep (forward any extra args to hparam_sweep.py)
uv run sweeps/hparam_sweep.py --trials_json "$SWEEP_TRIALS_JSON" "$@"
REMOTE