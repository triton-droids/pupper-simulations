#!/usr/bin/env bash
set -euo pipefail

# Always run relative to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env from script directory and export vars
set -a
source "$SCRIPT_DIR/.env"
set +a

echo "Starting deployment process..."
echo "Staging and committing changes"

cd "$SCRIPT_DIR"

git add .

git commit -m "Deploying latest changes for training" || echo "No changes to commit"
git push -u origin "$BRANCH_NAME"

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

ssh -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" bash -lc "
  set -euo pipefail

  cd \"$SSH_DIRECTORY/locomotion\"

  git fetch origin
  git checkout \"$BRANCH_NAME\"
  git pull --ff-only origin \"$BRANCH_NAME\"

  uv run train.py $*
"
