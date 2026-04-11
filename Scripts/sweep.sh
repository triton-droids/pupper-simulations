#!/usr/bin/env bash
set -euo pipefail

# Resolve project paths relative to this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from repo root.
set -a
source "$REPO_ROOT/.env"
set +a

# Always run git commands from the repo root.
cd "$REPO_ROOT"

echo "Starting deployment process for SWEEP..."
echo "Repo root: $REPO_ROOT"
echo "Staging and committing changes"

git add .
git commit -m "Deploying latest changes for sweep" || echo "No changes to commit"
git push -u origin "$BRANCH_NAME"

# Ensure local ssh-agent has GitHub key loaded (needed for -A forwarding).
if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
fi
ssh-add "$GITHUB_KEY_PATH" >/dev/null 2>&1 || true

# Which trials file to use on the remote. This path is relative to remote
# repo_root/locomotion because the remote script cd's into locomotion/.
SWEEP_TRIALS_JSON="${SWEEP_TRIALS_JSON:-sweeps/trials_2080ti_screen.json}"

# Local base output folder.
# Defaults to Scripts/outputs in the current repo layout.
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"

# How often to check for new artifacts (seconds).
SYNC_INTERVAL="${SYNC_INTERVAL:-15}"

# Run folder name.
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_NAME="Run_${SWEEP_ID}"

# Remote and local run paths.
REMOTE_SWEEP_REL="outputs/sweeps/${RUN_DIR_NAME}"
REMOTE_SWEEP_ABS="${SSH_DIRECTORY}/locomotion/${REMOTE_SWEEP_REL}"
LOCAL_SWEEP_BASE="${LOCAL_OUTPUT_DIR}/sweeps/${RUN_DIR_NAME}"

echo "Sweep ID: $SWEEP_ID"
echo "Run folder: $RUN_DIR_NAME"
echo "Remote sweep dir: $REMOTE_SWEEP_ABS"
echo "Local sweep dir: $LOCAL_SWEEP_BASE"
echo "Trials: $SWEEP_TRIALS_JSON"

mkdir -p "$LOCAL_SWEEP_BASE"

sync_artifacts() {
    local remote_sweep_abs="$1"
    local local_sweep_base="$2"

    mkdir -p "$local_sweep_base"

    ssh -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" \
    "test -d '$remote_sweep_abs' && find '$remote_sweep_abs' -type f \
    \( -name '*.mp4' \
    -o -name 'latest_metrics.json' \
    -o -name 'metrics_step_*.json' \
    -o -name 'latest_progress.png' \
    -o -name 'progress_step_*.png' \
    -o -name 'training_summary.json' \
    -o -name 'trial_result.json' \
    -o -name 'results.jsonl' \
    -o -name 'leaderboard.json' \
    -o -name 'best_trial.json' \) 2>/dev/null || true" |
    while read -r remote_file; do
        [ -n "$remote_file" ] || continue

        local rel_path="${remote_file#"$remote_sweep_abs"/}"
        local local_file="$local_sweep_base/$rel_path"
        local local_dir
        local base_name

        local_dir="$(dirname "$local_file")"
        base_name="$(basename "$remote_file")"

        mkdir -p "$local_dir"

        case "$base_name" in
            latest_metrics.json|latest_progress.png|training_summary.json|trial_result.json|results.jsonl|leaderboard.json|best_trial.json|latest_video.mp4)
                scp -P "$SSH_PORT" -i "$SSH_KEY_PATH" \
                    "$DROIDS_IP_ADDRESS:$remote_file" "$local_file" >/dev/null 2>&1 || true
                ;;
            *)
                if [ ! -f "$local_file" ]; then
                    scp -P "$SSH_PORT" -i "$SSH_KEY_PATH" \
                        "$DROIDS_IP_ADDRESS:$remote_file" "$local_file" >/dev/null 2>&1 || true
                fi
                ;;
        esac
    done

    date +"Last sync: %Y-%m-%d %H:%M:%S" > "$local_sweep_base/.last_sync.txt"
}

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

# Start background sync loop.
(
  while true; do
    sync_artifacts "$REMOTE_SWEEP_ABS" "$LOCAL_SWEEP_BASE"
    sleep "$SYNC_INTERVAL"
  done
) &
SYNC_PID=$!

cleanup() {
  echo ""
  echo "[sync] Stopping background sync..."
  kill "$SYNC_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh -A -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" bash -s -- \
  "$BRANCH_NAME" "$SSH_DIRECTORY" "$GITHUB_REPO_SSH" "$SWEEP_TRIALS_JSON" "$REMOTE_SWEEP_REL" "$@" <<'REMOTE'
set -euo pipefail

BRANCH_NAME="$1"; shift
SSH_DIRECTORY="$1"; shift
GITHUB_REPO_SSH="$1"; shift
SWEEP_TRIALS_JSON="$1"; shift
REMOTE_SWEEP_REL="$1"; shift

GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -n1 | cut -d, -f1)"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo "Using GPU $CUDA_VISIBLE_DEVICES"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

mkdir -p "$SSH_DIRECTORY"
cd "$SSH_DIRECTORY"

if [ ! -d .git ]; then
  git clone "$GITHUB_REPO_SSH" .
fi

git fetch origin

# Discard any leftover tracked-file edits from previous runs on the remote box.
git reset --hard
git clean -fd

# Force the working tree onto the requested branch and exact remote commit.
git checkout -B "$BRANCH_NAME" "origin/$BRANCH_NAME"
git reset --hard "origin/$BRANCH_NAME"
git clean -fd

cd locomotion

echo "Running hyperparameter sweep..."
echo "Trials: $SWEEP_TRIALS_JSON"
echo "Sweep output dir: $REMOTE_SWEEP_REL"

uv run sweeps/hparam_sweep.py --trials_json "$SWEEP_TRIALS_JSON" --base_output_dir "$REMOTE_SWEEP_REL" "$@"
REMOTE

# Final sync to catch artifacts produced near the end.
sync_artifacts "$REMOTE_SWEEP_ABS" "$LOCAL_SWEEP_BASE"

echo ""
echo "Sweep finished. Artifacts should now be synced under:"
echo "  $LOCAL_SWEEP_BASE"