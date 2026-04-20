#!/usr/bin/env bash

# Change ONLY CURRENT_TASK to switch which task this sweep runs.
TASK_DANCE_ENV_FILE="tasks/bittle_dance_env.py"
TASK_WALK_ENV_FILE="tasks/bittle_walk_env.py"
CURRENT_TASK="Dance"

set -euo pipefail

# Figure out where this script lives and where the repo root is.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Load reusable settings like SSH credentials and branch names from `.env`.
set -a
source <(tr -d '\r' < "$ENV_FILE")
set +a

SSH_COMMON_ARGS=(-o BatchMode=yes -o ConnectTimeout=15 -o Port="$SSH_PORT" -i "$SSH_KEY_PATH")

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

# Decide how the remote machine should use its GPUs.
#   single_gpu      = one visible GPU, trials one after another
#   multi_gpu       = one trial at a time, but that trial can see several GPUs
#   parallel_trials = several child trials at once, one GPU slot per child
SWEEP_REMOTE_MODE="${SWEEP_REMOTE_MODE:-parallel_trials}"

# Optionally pin the sweep to specific GPUs. "auto" means the remote box should
# choose the least-busy ones on its own.
SWEEP_GPU_IDS="${SWEEP_GPU_IDS:-auto}"

# If GPU choice is automatic, this says how many GPUs to reserve. "all" means
# use every visible GPU the remote box reports.
SWEEP_GPU_COUNT="${SWEEP_GPU_COUNT:-all}"

# In parallel mode, this says how many trials may run at once. "auto" means
# match the number of selected GPU slots.
SWEEP_PARALLEL_TRIALS="${SWEEP_PARALLEL_TRIALS:-auto}"

# Give each sweep one stable numbered label so both the remote artifacts and
# the local mirror use the same human-readable folder name.
SWEEP_NUMBER="$(allocate_sweep_number "$SWEEP_COUNTER_FILE")"
LOCAL_SWEEP_LABEL="Sweep_${SWEEP_NUMBER}"

# Build the matching remote and local folder paths for this sweep.
REMOTE_SWEEP_REL="../Scripts/Outputs/${LOCAL_SWEEP_LABEL}"
REMOTE_SWEEP_ABS="${SSH_DIRECTORY}/Scripts/Outputs/${LOCAL_SWEEP_LABEL}"
LOCAL_SWEEP_BASE="${LOCAL_OUTPUT_DIR}/${LOCAL_SWEEP_LABEL}"
LOCAL_SUBMISSION_RECEIPT="${LOCAL_SWEEP_BASE}/submission_receipt.txt"

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

write_local_submission_receipt() {
  cat > "$LOCAL_SUBMISSION_RECEIPT" <<EOF
local_sweep_label=$LOCAL_SWEEP_LABEL
task_name=$SELECTED_TASK_NAME
task_env_file=$SELECTED_TASK_ENV_FILE
trainer_trials_json=$SWEEP_TRIALS_JSON
task_hparams_json=$TASK_HPARAMS_JSON
remote_mode=$SWEEP_REMOTE_MODE
requested_gpu_ids=$SWEEP_GPU_IDS
requested_gpu_count=$SWEEP_GPU_COUNT
requested_parallel_trials=$SWEEP_PARALLEL_TRIALS
remote_sweep_abs=$REMOTE_SWEEP_ABS
local_output_dir=$LOCAL_SWEEP_BASE
submitted_at_local=$(date +"%Y-%m-%dT%H:%M:%S%z")
EOF
}

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

ssh -A "${SSH_COMMON_ARGS[@]}" "$DROIDS_IP_ADDRESS" bash -s -- \
  "$BRANCH_NAME" "$SSH_DIRECTORY" "$GITHUB_REPO_SSH" "$SWEEP_TRIALS_JSON" "$REMOTE_SWEEP_REL" "$REMOTE_SWEEP_ABS" "$LOCAL_SWEEP_LABEL" \
  "$SWEEP_REMOTE_MODE" "$SWEEP_GPU_IDS" "$SWEEP_GPU_COUNT" "$SWEEP_PARALLEL_TRIALS" \
  "$TASK_HPARAMS_JSON" "$SELECTED_TASK_NAME" "$SELECTED_TASK_ENV_FILE" "$@" <<'REMOTE'
set -euo pipefail

BRANCH_NAME="$1"; shift
SSH_DIRECTORY="$1"; shift
GITHUB_REPO_SSH="$1"; shift
SWEEP_TRIALS_JSON="$1"; shift
REMOTE_SWEEP_REL="$1"; shift
REMOTE_SWEEP_ABS="$1"; shift
LOCAL_SWEEP_LABEL="$1"; shift
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

  # Otherwise, pick the least-busy GPUs by current memory use. "all" keeps
  # every visible GPU instead of truncating the list.
  if [ "$requested_count" = "all" ]; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      sort -t, -k2 -n |
      cut -d, -f1 |
      tr -d ' ' |
      paste -sd, -
  else
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      sort -t, -k2 -n |
      head -n "$requested_count" |
      cut -d, -f1 |
      tr -d ' ' |
      paste -sd, -
  fi
}

count_csv_items() {
  local csv="$1"
  if [ -z "$csv" ]; then
    echo "0"
    return
  fi
  awk -F',' '{print NF}' <<< "$csv"
}

GPU_IDS="$(pick_gpu_ids "$SWEEP_GPU_COUNT")"
echo "Selected GPU ids: ${GPU_IDS:-<none>}"
GPU_SLOT_COUNT="$(count_csv_items "$GPU_IDS")"
if [ "$SWEEP_PARALLEL_TRIALS" = "auto" ]; then
  MAX_PARALLEL_TRIALS="$GPU_SLOT_COUNT"
else
  MAX_PARALLEL_TRIALS="$SWEEP_PARALLEL_TRIALS"
fi
if [ "$GPU_SLOT_COUNT" -lt 1 ]; then
  echo "ERROR: no GPU slots were selected for the sweep." >&2
  exit 2
fi
echo "Resolved GPU slot count: ${GPU_SLOT_COUNT}"
echo "Resolved parallel trials: ${MAX_PARALLEL_TRIALS}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

# Make sure the remote repo exists, is up to date, and matches the requested
# branch before we detach. This part still runs inside the live SSH session so
# it can use agent forwarding for any private Git access.
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

mkdir -p "$REMOTE_SWEEP_ABS"
STATUS_FILE="$REMOTE_SWEEP_ABS/.remote_status"
EXIT_CODE_FILE="$REMOTE_SWEEP_ABS/.remote_exit_code"
RUNNER_SCRIPT="$REMOTE_SWEEP_ABS/run_remote_sweep.sh"
RUNNER_LOG="$REMOTE_SWEEP_ABS/remote_runner.log"
REMOTE_PID_FILE="$REMOTE_SWEEP_ABS/.remote_pid"
JOB_INFO_FILE="$REMOTE_SWEEP_ABS/job_info.txt"

rm -f "$EXIT_CODE_FILE" "$REMOTE_PID_FILE" "$RUNNER_LOG"
printf 'launching\n' > "$STATUS_FILE"

cat > "$JOB_INFO_FILE" <<EOF
local_sweep_label=$LOCAL_SWEEP_LABEL
task_name=$SELECTED_TASK_NAME
task_env_file=$SELECTED_TASK_ENV_FILE
trainer_trials_json=$SWEEP_TRIALS_JSON
task_hparams_json=$TASK_HPARAMS_JSON
remote_mode=$SWEEP_REMOTE_MODE
requested_gpu_ids=$SWEEP_GPU_IDS
requested_gpu_count=$SWEEP_GPU_COUNT
resolved_gpu_ids=$GPU_IDS
gpu_slot_count=$GPU_SLOT_COUNT
resolved_parallel_trials=$MAX_PARALLEL_TRIALS
branch_name=$BRANCH_NAME
remote_sweep_abs=$REMOTE_SWEEP_ABS
submitted_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

cat > "$RUNNER_SCRIPT" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

SSH_DIRECTORY="$1"; shift
SWEEP_TRIALS_JSON="$1"; shift
REMOTE_SWEEP_REL="$1"; shift
SWEEP_REMOTE_MODE="$1"; shift
GPU_IDS="$1"; shift
MAX_PARALLEL_TRIALS="$1"; shift
TASK_HPARAMS_JSON="$1"; shift
SELECTED_TASK_NAME="$1"; shift
SELECTED_TASK_ENV_FILE="$1"; shift
STATUS_FILE="$1"; shift
EXIT_CODE_FILE="$1"; shift

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

cleanup_remote_status() {
  local exit_code="$1"
  printf '%s\n' "$exit_code" > "$EXIT_CODE_FILE"
  if [ "$exit_code" -eq 0 ]; then
    printf 'succeeded\n' > "$STATUS_FILE"
  else
    printf 'failed\n' > "$STATUS_FILE"
  fi
}

trap 'cleanup_remote_status "$?"' EXIT

printf 'running\n' > "$STATUS_FILE"

cd "$SSH_DIRECTORY/locomotion"

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
      --max_concurrent_trials "$MAX_PARALLEL_TRIALS" \
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
RUNNER

chmod +x "$RUNNER_SCRIPT"

nohup bash "$RUNNER_SCRIPT" \
  "$SSH_DIRECTORY" \
  "$SWEEP_TRIALS_JSON" \
  "$REMOTE_SWEEP_REL" \
  "$SWEEP_REMOTE_MODE" \
  "$GPU_IDS" \
  "$MAX_PARALLEL_TRIALS" \
  "$TASK_HPARAMS_JSON" \
  "$SELECTED_TASK_NAME" \
  "$SELECTED_TASK_ENV_FILE" \
  "$STATUS_FILE" \
  "$EXIT_CODE_FILE" \
  "$@" > "$RUNNER_LOG" 2>&1 < /dev/null &

REMOTE_PID="$!"
printf '%s\n' "$REMOTE_PID" > "$REMOTE_PID_FILE"

echo "Detached remote sweep launched successfully."
echo "Remote PID: $REMOTE_PID"
echo "Selected GPU ids: ${GPU_IDS:-<none>}"
echo "Resolved GPU slot count: ${GPU_SLOT_COUNT}"
echo "Resolved parallel trials: ${MAX_PARALLEL_TRIALS}"
echo "Remote status file: $STATUS_FILE"
echo "Remote log file: $RUNNER_LOG"
echo "Remote job info file: $JOB_INFO_FILE"
REMOTE

write_local_submission_receipt

echo ""
echo "Job sent to remote server."
echo "Remote sweep dir:"
echo "  $REMOTE_SWEEP_ABS"
echo "Local receipt:"
echo "  $LOCAL_SUBMISSION_RECEIPT"
echo "Use Scripts/retrieve_sweeps.sh when you want to pull results back."
