#!/usr/bin/env bash
set -euo pipefail

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

# Which trials file to use on the remote
SWEEP_TRIALS_JSON="${SWEEP_TRIALS_JSON:-sweeps/trials_2080ti_screen.json}"

# Local destination for downloaded videos and artifacts
# On Windows Git Bash, use /c/Users/... not C:\Users\...
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-/c/Users/brand/PycharmProjects/pupper-simulations/outputs}"

# How often to check for new videos (seconds)
SYNC_INTERVAL="${SYNC_INTERVAL:-15}"

# Pin the sweep output dir name so we know exactly where to look
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
REMOTE_SWEEP_REL="outputs/sweeps/sweep_${SWEEP_ID}"
REMOTE_SWEEP_ABS="${SSH_DIRECTORY}/locomotion/${REMOTE_SWEEP_REL}"

echo "Sweep ID: $SWEEP_ID"
echo "Remote sweep dir: $REMOTE_SWEEP_ABS"
echo "Local output dir: $LOCAL_OUTPUT_DIR"
echo "Trials: $SWEEP_TRIALS_JSON"

mkdir -p "$LOCAL_OUTPUT_DIR"

sync_videos() {
  echo "[sync] Starting background sync loop..."
  local seen_file="$LOCAL_OUTPUT_DIR/.synced_${SWEEP_ID}.txt"
  touch "$seen_file"

  while true; do
    # List mp4 files on remote, if the folder exists yet
    remote_list="$(
      ssh -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" \
        "test -d '$REMOTE_SWEEP_ABS' && find '$REMOTE_SWEEP_ABS' -type f -name '*.mp4' 2>/dev/null || true" \
      || true
    )"

    if [[ -n "$remote_list" ]]; then
      while IFS= read -r rfile; do
        [[ -z "$rfile" ]] && continue
        # Skip if already synced
        if grep -Fxq "$rfile" "$seen_file"; then
          continue
        fi

        # Preserve relative path under LOCAL_OUTPUT_DIR
        rel="${rfile#${SSH_DIRECTORY}/locomotion/}"
        local_path="$LOCAL_OUTPUT_DIR/$rel"
        mkdir -p "$(dirname "$local_path")"

        echo "[sync] Downloading: $rel"
        scp -P "$SSH_PORT" -i "$SSH_KEY_PATH" \
          "$DROIDS_IP_ADDRESS:$rfile" "$local_path" >/dev/null 2>&1 || true

        # Mark as synced if it exists locally
        if [[ -f "$local_path" ]]; then
          echo "$rfile" >> "$seen_file"
        fi
      done <<< "$remote_list"
    fi

    sleep "$SYNC_INTERVAL"
  done
}

# Start background sync
sync_videos &
SYNC_PID=$!

cleanup() {
  echo ""
  echo "[sync] Stopping background sync..."
  kill "$SYNC_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

ssh -A -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" bash -s -- \
  "$BRANCH_NAME" "$SSH_DIRECTORY" "$GITHUB_REPO_SSH" "$SWEEP_TRIALS_JSON" "$REMOTE_SWEEP_REL" "$@" <<'REMOTE'
set -euo pipefail

BRANCH_NAME="$1"; shift
SSH_DIRECTORY="$1"; shift
GITHUB_REPO_SSH="$1"; shift
SWEEP_TRIALS_JSON="$1"; shift
REMOTE_SWEEP_REL="$1"; shift

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
echo "Sweep output dir: $REMOTE_SWEEP_REL"

uv run sweeps/hparam_sweep.py --trials_json "$SWEEP_TRIALS_JSON" --base_output_dir "$REMOTE_SWEEP_REL" "$@"
REMOTE

echo ""
echo "Sweep finished. Videos should now be synced under:"
echo "  $LOCAL_OUTPUT_DIR/$REMOTE_SWEEP_REL"