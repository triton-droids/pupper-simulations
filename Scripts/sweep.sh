#!/usr/bin/env bash

# Change ONLY CURRENT_TASK to switch which task this sweep runs.
TASK_DANCE_ENV_FILE="tasks/bittle_dance_env.py"
TASK_WALK_ENV_FILE="tasks/bittle_walk_env.py"
CURRENT_TASK="Dance"

set -euo pipefail

# Figure out where this script lives and where the repo root is.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load reusable settings like SSH credentials and branch names from `.env`.
set -a
source "$REPO_ROOT/.env"
set +a

# Run all git commands from the repo root so paths stay predictable.
cd "$REPO_ROOT"

case "$CURRENT_TASK" in
  Dance|dance)
    SELECTED_TASK_NAME="dance"
    SELECTED_TASK_ENV_FILE="$TASK_DANCE_ENV_FILE"
    TASK_HPARAMS_JSON="tasks/bittle_dance_hparams.json"
    ;;
  Walk|walk)
    SELECTED_TASK_NAME="walking"
    SELECTED_TASK_ENV_FILE="$TASK_WALK_ENV_FILE"
    TASK_HPARAMS_JSON="tasks/bittle_walking_hparams.json"
    ;;
  *)
    echo "ERROR: CURRENT_TASK must be Dance or Walk, not '$CURRENT_TASK'" >&2
    exit 2
    ;;
esac

if [ ! -f "$REPO_ROOT/locomotion/$SELECTED_TASK_ENV_FILE" ]; then
  echo "ERROR: selected task file was not found: $REPO_ROOT/locomotion/$SELECTED_TASK_ENV_FILE" >&2
  exit 2
fi

if [ ! -f "$REPO_ROOT/locomotion/$TASK_HPARAMS_JSON" ]; then
  echo "ERROR: selected task hyperparameter JSON was not found: $REPO_ROOT/locomotion/$TASK_HPARAMS_JSON" >&2
  exit 2
fi

echo "Starting deployment process for SWEEP..."
echo "Repo root: $REPO_ROOT"
echo "Selected task label: $CURRENT_TASK"
echo "Selected task name: $SELECTED_TASK_NAME"
echo "Selected task script: $SELECTED_TASK_ENV_FILE"
echo "Staging and committing changes"

git add .
git commit -m "Deploying latest changes for sweep" || echo "No changes to commit"
git push -u origin "$BRANCH_NAME"

# Make sure the local SSH agent has the GitHub key so the remote machine can
# reuse it when pulling the repo.
if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
fi
ssh-add "$GITHUB_KEY_PATH" >/dev/null 2>&1 || true

# Choose which JSON file describes the trainer-side PPO sweep. The remote
# script later runs from inside `locomotion/`, so this path is relative to
# that folder.
SWEEP_TRIALS_JSON="${SWEEP_TRIALS_JSON:-sweeps/training_budget_and_batching_sweep.json}"

# Choose where copied-back artifacts should land on the local machine. By
# default, keep them under `Scripts/Outputs/`.
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$REPO_ROOT/Scripts/Outputs}"

# Keep a tiny local counter so each sweep gets a human-readable folder like
# `Sweep #0`, `Sweep #1`, and so on, based on how many times this script has
# been invoked on this machine.
SWEEP_COUNTER_FILE="${SWEEP_COUNTER_FILE:-$REPO_ROOT/Scripts/.sweep_counter}"

allocate_sweep_number() {
  local counter_file="$1"
  local current_number="0"

  if [ -f "$counter_file" ]; then
    current_number="$(tr -d '[:space:]' < "$counter_file")"
    if ! [[ "$current_number" =~ ^[0-9]+$ ]]; then
      echo "WARNING: invalid sweep counter '$current_number'; resetting to 0" >&2
      current_number="0"
    fi
  fi

  printf '%s\n' "$((current_number + 1))" > "$counter_file"
  printf '%s' "$current_number"
}

# Decide how often to poll the remote machine for newly finished artifacts.
SYNC_INTERVAL="${SYNC_INTERVAL:-15}"

# Decide how the remote machine should use its GPUs.
#   single_gpu      = one visible GPU, trials one after another
#   multi_gpu       = one trial at a time, but that trial can see several GPUs
#   parallel_trials = several child trials at once, one GPU slot per child
SWEEP_REMOTE_MODE="${SWEEP_REMOTE_MODE:-single_gpu}"

# Optionally pin the sweep to specific GPUs. "auto" means the remote box should
# choose the least-busy ones on its own.
SWEEP_GPU_IDS="${SWEEP_GPU_IDS:-auto}"

# If GPU choice is automatic, this says how many GPUs to reserve.
SWEEP_GPU_COUNT="${SWEEP_GPU_COUNT:-1}"

# In parallel mode, this says how many trials may run at once.
SWEEP_PARALLEL_TRIALS="${SWEEP_PARALLEL_TRIALS:-$SWEEP_GPU_COUNT}"

# Give each sweep one stable numbered label so both the remote artifacts and
# the local mirror use the same human-readable folder name.
SWEEP_NUMBER="$(allocate_sweep_number "$SWEEP_COUNTER_FILE")"
LOCAL_SWEEP_LABEL="Sweep_${SWEEP_NUMBER}"

# Build the matching remote and local folder paths for this sweep.
REMOTE_SWEEP_REL="../Scripts/Outputs/${LOCAL_SWEEP_LABEL}"
REMOTE_SWEEP_ABS="${SSH_DIRECTORY}/Scripts/Outputs/${LOCAL_SWEEP_LABEL}"
LOCAL_SWEEP_BASE="${LOCAL_OUTPUT_DIR}/${LOCAL_SWEEP_LABEL}"

echo "Sweep number: $SWEEP_NUMBER"
echo "Local sweep label: $LOCAL_SWEEP_LABEL"
echo "Remote sweep dir: $REMOTE_SWEEP_ABS"
echo "Local sweep dir: $LOCAL_SWEEP_BASE"
echo "Trials: $SWEEP_TRIALS_JSON"
echo "Task hyperparameters: $TASK_HPARAMS_JSON"
echo "Remote mode: $SWEEP_REMOTE_MODE"
echo "GPU ids: $SWEEP_GPU_IDS"
echo "GPU count: $SWEEP_GPU_COUNT"
echo "Parallel trials: $SWEEP_PARALLEL_TRIALS"

mkdir -p "$LOCAL_SWEEP_BASE"

sync_artifacts() {
    local remote_sweep_abs="$1"
    local local_sweep_base="$2"

    # Make sure the local destination exists before we start copying files in.
    mkdir -p "$local_sweep_base"

    # Ask the remote host for the current list of artifact files we care about.
    ssh -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" \
    "test -d '$remote_sweep_abs' && find '$remote_sweep_abs' -type f \
    \( -name '*.mp4' \
    -o -name 'final_metrics.json' \
    -o -name 'final_progress.png' \
    -o -name 'training_summary.json' \
    -o -name 'trial_result.json' \
    -o -name 'results.jsonl' \
    -o -name 'leaderboard.json' \
    -o -name 'best_trial.json' \) 2>/dev/null || true" |
    while read -r remote_file; do
        [ -n "$remote_file" ] || continue

        # Recreate the same relative folder layout locally.
        local rel_path="${remote_file#"$remote_sweep_abs"/}"
        local local_file="$local_sweep_base/$rel_path"
        local local_dir
        local base_name

        local_dir="$(dirname "$local_file")"
        base_name="$(basename "$remote_file")"

        mkdir -p "$local_dir"

        # Always refresh the "latest final" files, but avoid recopying older
        # immutable files unnecessarily.
        case "$base_name" in
            final_metrics.json|final_progress.png|training_summary.json|trial_result.json|results.jsonl|leaderboard.json|best_trial.json|final_video.mp4)
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

    # Drop a small timestamp file so you can tell when the last sync happened.
    date +"Last sync: %Y-%m-%d %H:%M:%S" > "$local_sweep_base/.last_sync.txt"
}

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

# Keep syncing in the background while the remote sweep is running.
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
  "$BRANCH_NAME" "$SSH_DIRECTORY" "$GITHUB_REPO_SSH" "$SWEEP_TRIALS_JSON" "$REMOTE_SWEEP_REL" \
  "$SWEEP_REMOTE_MODE" "$SWEEP_GPU_IDS" "$SWEEP_GPU_COUNT" "$SWEEP_PARALLEL_TRIALS" \
  "$TASK_HPARAMS_JSON" "$SELECTED_TASK_NAME" "$SELECTED_TASK_ENV_FILE" "$@" <<'REMOTE'
set -euo pipefail

BRANCH_NAME="$1"; shift
SSH_DIRECTORY="$1"; shift
GITHUB_REPO_SSH="$1"; shift
SWEEP_TRIALS_JSON="$1"; shift
REMOTE_SWEEP_REL="$1"; shift
SWEEP_REMOTE_MODE="$1"; shift
SWEEP_GPU_IDS="$1"; shift
SWEEP_GPU_COUNT="$1"; shift
SWEEP_PARALLEL_TRIALS="$1"; shift
TASK_HPARAMS_JSON="$1"; shift
SELECTED_TASK_NAME="$1"; shift
SELECTED_TASK_ENV_FILE="$1"; shift

pick_gpu_ids() {
  local requested_count="$1"

  # If the caller named specific GPUs, use those exactly.
  if [ -n "$SWEEP_GPU_IDS" ] && [ "$SWEEP_GPU_IDS" != "auto" ]; then
    echo "$SWEEP_GPU_IDS"
    return
  fi

  # Otherwise, pick the least-busy GPUs by current memory use.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    sort -t, -k2 -n |
    head -n "$requested_count" |
    cut -d, -f1 |
    tr -d ' ' |
    paste -sd, -
}

GPU_IDS="$(pick_gpu_ids "$SWEEP_GPU_COUNT")"
echo "Selected GPU ids: ${GPU_IDS:-<none>}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

# Make sure the remote repo exists, is up to date, and matches the requested branch.
mkdir -p "$SSH_DIRECTORY"
cd "$SSH_DIRECTORY"

if [ ! -d .git ]; then
  git clone "$GITHUB_REPO_SSH" .
fi

git fetch origin

# Clear any old remote working-tree leftovers so the sweep starts from a clean checkout.
git reset --hard
git clean -fd

# Force the remote checkout to match the current branch tip exactly.
git checkout -B "$BRANCH_NAME" "origin/$BRANCH_NAME"
git reset --hard "origin/$BRANCH_NAME"
git clean -fd

cd locomotion

echo "Running hyperparameter sweep..."
echo "Task name: $SELECTED_TASK_NAME"
echo "Task env script: $SELECTED_TASK_ENV_FILE"
echo "Trials: $SWEEP_TRIALS_JSON"
echo "Task hyperparameters: $TASK_HPARAMS_JSON"
echo "Sweep output dir: $REMOTE_SWEEP_REL"
echo "Remote mode: $SWEEP_REMOTE_MODE"

TASK_HPARAMS_ARGS=()
if [ -n "$TASK_HPARAMS_JSON" ]; then
  TASK_HPARAMS_ARGS+=(--task_hparams_json "$TASK_HPARAMS_JSON")
fi

case "$SWEEP_REMOTE_MODE" in
  single_gpu)
    # One visible GPU, one trial at a time.
    export CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}"
    echo "Using single GPU: $CUDA_VISIBLE_DEVICES"
    uv run sweeps/hparam_sweep.py \
      --trials_json "$SWEEP_TRIALS_JSON" \
      --base_output_dir "$REMOTE_SWEEP_REL" \
      --task "$SELECTED_TASK_NAME" \
      "${TASK_HPARAMS_ARGS[@]}" \
      "$@"
    ;;
  multi_gpu)
    # One training job at a time, but let that job see several GPUs.
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Using multi-GPU run with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    uv run sweeps/hparam_sweep.py \
      --trials_json "$SWEEP_TRIALS_JSON" \
      --base_output_dir "$REMOTE_SWEEP_REL" \
      --task "$SELECTED_TASK_NAME" \
      "${TASK_HPARAMS_ARGS[@]}" \
      "$@"
    ;;
  parallel_trials)
    # Many child trials at once, each assigned to one reserved GPU slot.
    echo "Running concurrent trials across GPUs: $GPU_IDS"
    uv run sweeps/hparam_sweep.py \
      --trials_json "$SWEEP_TRIALS_JSON" \
      --base_output_dir "$REMOTE_SWEEP_REL" \
      --task "$SELECTED_TASK_NAME" \
      --max_concurrent_trials "$SWEEP_PARALLEL_TRIALS" \
      --available_gpus "$GPU_IDS" \
      "${TASK_HPARAMS_ARGS[@]}" \
      "$@"
    ;;
  *)
    echo "ERROR: unknown SWEEP_REMOTE_MODE '$SWEEP_REMOTE_MODE'"
    echo "Expected one of: single_gpu, multi_gpu, parallel_trials"
    exit 2
    ;;
esac
REMOTE

# Do one last sync at the end so late-written files are not missed.
sync_artifacts "$REMOTE_SWEEP_ABS" "$LOCAL_SWEEP_BASE"

echo ""
echo "Sweep finished. Artifacts should now be synced under:"
echo "  $LOCAL_SWEEP_BASE"
