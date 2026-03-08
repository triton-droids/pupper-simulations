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

# Local base output folder
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-/c/Users/brand/PycharmProjects/pupper-simulations/outputs}"

# How often to check for new artifacts (seconds)
SYNC_INTERVAL="${SYNC_INTERVAL:-15}"

# Run folder name
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_NAME="Run_${SWEEP_ID}"

# Remote and local run paths
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
    local REMOTE_SWEEP_ABS="$1"
    local LOCAL_SWEEP_BASE="$2"

    mkdir -p "$LOCAL_SWEEP_BASE"

    ssh -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" \
    "test -d '$REMOTE_SWEEP_ABS' && find '$REMOTE_SWEEP_ABS' -type f \
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

        rel_path="${remote_file#"$REMOTE_SWEEP_ABS"/}"
        local_file="$LOCAL_SWEEP_BASE/$rel_path"
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

    date +"Last sync: %Y-%m-%d %H:%M:%S" > "$LOCAL_SWEEP_BASE/.last_sync.txt"
}

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

# Start background sync loop
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
git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"
git reset --hard "origin/$BRANCH_NAME"

cd locomotion

echo "Running hyperparameter sweep..."
echo "Trials: $SWEEP_TRIALS_JSON"
echo "Sweep output dir: $REMOTE_SWEEP_REL"

uv run sweeps/hparam_sweep.py --trials_json "$SWEEP_TRIALS_JSON" --base_output_dir "$REMOTE_SWEEP_REL" "$@"
REMOTE

# Final sync to catch artifacts produced near the end
sync_artifacts "$REMOTE_SWEEP_ABS" "$LOCAL_SWEEP_BASE"

echo ""
echo "Sweep finished. Artifacts should now be synced under:"
echo "  $LOCAL_SWEEP_BASE"