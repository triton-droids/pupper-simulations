#!/usr/bin/env bash
set -euo pipefail

# Load env vars from .env next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/.env"
set +a

echo "Starting deployment process..."
echo "Staging and committing changes"

git add .
git commit -m "Deploying latest changes for training" || echo "No changes to commit"
git push -u origin "$BRANCH_NAME"

# Ensure local ssh-agent has GitHub key loaded (needed for -A forwarding)
if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
fi
ssh-add "$GITHUB_KEY_PATH" >/dev/null 2>&1 || true

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

ssh -A -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" bash -s -- "$BRANCH_NAME" "$SSH_DIRECTORY" "$GITHUB_REPO_SSH" "$@" <<'REMOTE'
set -euo pipefail

BRANCH_NAME="$1"; shift
SSH_DIRECTORY="$1"; shift
GITHUB_REPO_SSH="$1"; shift

# Pick GPU and tame JAX memory behavior (adjust as needed)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"

mkdir -p "$SSH_DIRECTORY"
cd "$SSH_DIRECTORY"

if [ ! -d .git ]; then
  git clone "$GITHUB_REPO_SSH" .
fi

git fetch origin
git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"
git reset --hard "origin/$BRANCH_NAME"

cd locomotion

# Run training (forward args passed to train.sh)
uv run train.py "$@"
REMOTE
