#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

set -a
source <(tr -d '\r' < "$ENV_FILE")
set +a

SSH_COMMON_ARGS=(-o BatchMode=yes -o ConnectTimeout=15 -o Port="$SSH_PORT" -i "$SSH_KEY_PATH")

LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$REPO_ROOT/Scripts/Outputs}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-$SSH_DIRECTORY/Scripts/Outputs}"
DELETE_REMOTE_AFTER_DOWNLOAD="${DELETE_REMOTE_AFTER_DOWNLOAD:-true}"

mkdir -p "$LOCAL_OUTPUT_DIR"

normalize_field() {
  local value="${1:-}"
  if [ -z "$value" ]; then
    printf '?'
  else
    printf '%s' "$value"
  fi
}

list_remote_sweeps() {
  ssh "${SSH_COMMON_ARGS[@]}" "$DROIDS_IP_ADDRESS" bash -s -- "$REMOTE_OUTPUT_DIR" <<'REMOTE'
set -euo pipefail

REMOTE_OUTPUT_DIR="$1"

if [ ! -d "$REMOTE_OUTPUT_DIR" ]; then
  exit 0
fi

find "$REMOTE_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'Sweep_*' | sort -V | while read -r sweep_dir; do
  label="$(basename "$sweep_dir")"
  status="unknown"
  exit_code=""
  task_name=""
  submitted_at=""
  completed_trials="0"

  if [ -f "$sweep_dir/.remote_status" ]; then
    status="$(tr -d '\r\n' < "$sweep_dir/.remote_status")"
  fi

  if [ -f "$sweep_dir/.remote_exit_code" ]; then
    exit_code="$(tr -d '\r\n' < "$sweep_dir/.remote_exit_code")"
  fi

  if [ -f "$sweep_dir/job_info.txt" ]; then
    task_name="$(sed -n 's/^task_name=//p' "$sweep_dir/job_info.txt" | head -n 1)"
    submitted_at="$(sed -n 's/^submitted_at_utc=//p' "$sweep_dir/job_info.txt" | head -n 1)"
  fi

  if [ -f "$sweep_dir/results.jsonl" ]; then
    completed_trials="$(wc -l < "$sweep_dir/results.jsonl" | tr -d '[:space:]')"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$status" "$exit_code" "$task_name" "$submitted_at" "$completed_trials"
done
REMOTE
}

print_section() {
  local heading="$1"
  local match_mode="$2"
  local listing_file="$3"
  local found="0"

  echo "$heading"

  while IFS=$'\t' read -r label status exit_code task_name submitted_at completed_trials; do
    [ -n "${label:-}" ] || continue

    case "$match_mode" in
      finished)
        case "$status" in
          succeeded|failed) ;;
          *) continue ;;
        esac
        ;;
      running)
        case "$status" in
          launching|running|not_started|"") ;;
          *) continue ;;
        esac
        ;;
      other)
        case "$status" in
          succeeded|failed|launching|running|not_started|"") continue ;;
        esac
        ;;
      *)
        echo "Internal error: unknown section mode '$match_mode'" >&2
        exit 2
        ;;
    esac

    found="1"
    printf '  %s | status=%s | task=%s | submitted=%s | completed_trials=%s' \
      "$label" \
      "$(normalize_field "$status")" \
      "$(normalize_field "$task_name")" \
      "$(normalize_field "$submitted_at")" \
      "$(normalize_field "$completed_trials")"
    if [ -n "$exit_code" ]; then
      printf ' | exit_code=%s' "$exit_code"
    fi
    printf '\n'
  done < "$listing_file"

  if [ "$found" -eq 0 ]; then
    echo "  (none)"
  fi

  echo ""
}

find_selected_sweep() {
  local selected_label="$1"
  local listing_file="$2"

  while IFS=$'\t' read -r label status exit_code task_name submitted_at completed_trials; do
    [ -n "${label:-}" ] || continue
    if [ "$label" = "$selected_label" ]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$status" "$exit_code" "$task_name" "$submitted_at" "$completed_trials"
      return 0
    fi
  done < "$listing_file"

  return 1
}

should_refresh_file() {
  local base_name="$1"

  case "$base_name" in
    .remote_status|.remote_exit_code|.remote_pid|.last_sync.txt|job_info.txt|remote_runner.log|run_remote_sweep.sh|results.jsonl|leaderboard.json|best_trial.json|final_metrics.json|final_progress.png|final_video.mp4|training_summary.json|trial_result.json)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

sync_remote_sweep() {
  local selected_label="$1"
  local remote_sweep_abs="$REMOTE_OUTPUT_DIR/$selected_label"
  local local_sweep_dir="$LOCAL_OUTPUT_DIR/$selected_label"
  local remote_files=""

  mkdir -p "$local_sweep_dir"

  remote_files="$(ssh "${SSH_COMMON_ARGS[@]}" "$DROIDS_IP_ADDRESS" \
    "test -d '$remote_sweep_abs' && find '$remote_sweep_abs' -type f 2>/dev/null || true")"

  if [ -z "$remote_files" ]; then
    echo "No remote files were found yet for $selected_label." >&2
    return 1
  fi

  while IFS= read -r remote_file; do
    [ -n "$remote_file" ] || continue

    rel_path="${remote_file#"$remote_sweep_abs"/}"
    local_file="$local_sweep_dir/$rel_path"
    local_dir="$(dirname "$local_file")"
    base_name="$(basename "$remote_file")"

    mkdir -p "$local_dir"

    if should_refresh_file "$base_name" || [ ! -f "$local_file" ]; then
      scp "${SSH_COMMON_ARGS[@]}" \
        "$DROIDS_IP_ADDRESS:$remote_file" "$local_file" >/dev/null
    fi
  done <<< "$remote_files"

  date +"Last sync: %Y-%m-%d %H:%M:%S" > "$local_sweep_dir/.last_sync.txt"
}

delete_remote_sweep() {
  local selected_label="$1"

  ssh "${SSH_COMMON_ARGS[@]}" "$DROIDS_IP_ADDRESS" bash -s -- \
    "$REMOTE_OUTPUT_DIR" "$selected_label" <<'REMOTE'
set -euo pipefail

REMOTE_OUTPUT_DIR="$1"
SELECTED_LABEL="$2"

case "$SELECTED_LABEL" in
  Sweep_[0-9]*)
    ;;
  *)
    echo "ERROR: refusing to delete unexpected sweep label '$SELECTED_LABEL'." >&2
    exit 2
    ;;
esac

REMOTE_SWEEP_ABS="$REMOTE_OUTPUT_DIR/$SELECTED_LABEL"

if [ ! -d "$REMOTE_SWEEP_ABS" ]; then
  echo "ERROR: remote sweep directory was not found: $REMOTE_SWEEP_ABS" >&2
  exit 2
fi

case "$REMOTE_SWEEP_ABS" in
  "$REMOTE_OUTPUT_DIR"/Sweep_[0-9]*)
    ;;
  *)
    echo "ERROR: refusing to delete unexpected remote path '$REMOTE_SWEEP_ABS'." >&2
    exit 2
    ;;
esac

rm -rf -- "$REMOTE_SWEEP_ABS"
REMOTE
}

REMOTE_LISTING_FILE="$(mktemp)"
trap 'rm -f "$REMOTE_LISTING_FILE"' EXIT

echo "Checking remote sweeps on $DROIDS_IP_ADDRESS ..."
list_remote_sweeps > "$REMOTE_LISTING_FILE"

if [ ! -s "$REMOTE_LISTING_FILE" ]; then
  echo "No remote sweeps were found under $REMOTE_OUTPUT_DIR"
  exit 0
fi

echo ""
print_section "Finished sweeps:" "finished" "$REMOTE_LISTING_FILE"
print_section "Running sweeps:" "running" "$REMOTE_LISTING_FILE"
print_section "Other sweeps:" "other" "$REMOTE_LISTING_FILE"

SELECTED_LABEL="${1:-}"

if [ -z "$SELECTED_LABEL" ]; then
  read -r -p "Type the sweep label to download (for example Sweep_0), or press Enter to cancel: " SELECTED_LABEL
fi

if [ -z "$SELECTED_LABEL" ]; then
  echo "No sweep selected."
  exit 0
fi

if ! SELECTED_ROW="$(find_selected_sweep "$SELECTED_LABEL" "$REMOTE_LISTING_FILE")"; then
  echo "ERROR: sweep '$SELECTED_LABEL' was not found on the remote server." >&2
  exit 2
fi

IFS=$'\t' read -r _ SELECTED_STATUS SELECTED_EXIT_CODE SELECTED_TASK_NAME SELECTED_SUBMITTED_AT SELECTED_COMPLETED_TRIALS <<< "$SELECTED_ROW"

echo ""
echo "Downloading $SELECTED_LABEL ..."
echo "  status: $(normalize_field "$SELECTED_STATUS")"
echo "  task: $(normalize_field "$SELECTED_TASK_NAME")"
echo "  submitted: $(normalize_field "$SELECTED_SUBMITTED_AT")"
echo "  completed trials so far: $(normalize_field "$SELECTED_COMPLETED_TRIALS")"

if [ "$SELECTED_STATUS" != "succeeded" ] && [ "$SELECTED_STATUS" != "failed" ]; then
  echo "  note: this sweep is still running or not finalized; the download will be a snapshot of what exists so far."
fi

sync_remote_sweep "$SELECTED_LABEL"

echo ""
echo "Download finished."
echo "Local sweep folder:"
echo "  $LOCAL_OUTPUT_DIR/$SELECTED_LABEL"

if [ "$DELETE_REMOTE_AFTER_DOWNLOAD" = "true" ]; then
  if [ "$SELECTED_STATUS" = "succeeded" ] || [ "$SELECTED_STATUS" = "failed" ]; then
    echo "Deleting remote sweep folder..."
    delete_remote_sweep "$SELECTED_LABEL"
    echo "Remote sweep deleted:"
    echo "  $REMOTE_OUTPUT_DIR/$SELECTED_LABEL"
  else
    echo "Remote sweep kept because it is still running or not finalized."
  fi
else
  echo "Remote sweep kept because DELETE_REMOTE_AFTER_DOWNLOAD=$DELETE_REMOTE_AFTER_DOWNLOAD."
fi
