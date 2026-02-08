#!/usr/bin/env bash
set -euo pipefail

# Load env vars from .env next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/.env"
set +a

echo "Starting deployment process..."

# 1) Stage, commit, and push current working changes
echo "Staging and committing changes"

git add .
git commit -m "Deploying latest changes for training" || echo "No changes to commit"
git push -u origin "$BRANCH_NAME"

# 2) Ensure local ssh-agent has GitHub key loaded (needed for -A forwarding)
if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
fi

# Only add if not already present
if ! ssh-add -l 2>/dev/null | grep -q "$(basename "$GITHUB_KEY_PATH")"; then
  ssh-add "$GITHUB_KEY_PATH"
fi

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

# 3) SSH into remote and run training
ssh -A -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" bash -lc "
  set -euo pipefail

  mkdir -p \"$SSH_DIRECTORY\"
  cd \"$SSH_DIRECTORY\"

  # If this directory is not a git repo yet, clone into it
  if [ ! -d .git ]; then
    git clone \"$GITHUB_REPO_SSH\" .
  fi

  git fetch origin
  git checkout \"$BRANCH_NAME\" || git checkout -b \"$BRANCH_NAME\" \"origin/$BRANCH_NAME\"
  git reset --hard \"origin/$BRANCH_NAME\"

  cd locomotion

  # Run training (forward any args you pass to train.sh)
  uv run train.py \"\$@\"
" _ "$@"
