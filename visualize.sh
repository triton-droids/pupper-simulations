#!/usr/bin/env bash
# Download trained policy from remote SSH node and launch MuJoCo teleop.
#
# Usage:
#   ./visualize.sh               # download policy + launch teleop
#   ./visualize.sh --dry-run      # print commands without executing
#   ./visualize.sh --download-only # download only, no teleop
#   ./visualize.sh --video        # also download training video
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# ── Flags ──────────────────────────────────────────────────────────────
DRY_RUN=false
DOWNLOAD_ONLY=false
DOWNLOAD_VIDEO=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=true ;;
    --download-only) DOWNLOAD_ONLY=true ;;
    --video)         DOWNLOAD_VIDEO=true ;;
    *)
      echo "Error: Unknown flag '$arg'"
      echo ""
      echo "Usage: ./visualize.sh [--dry-run] [--download-only] [--video]"
      echo "  --dry-run        Print commands without executing"
      echo "  --download-only  Download policy but don't launch teleop"
      echo "  --video          Also download training video"
      exit 1
      ;;
  esac
done

# ── Load .env ──────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: .env file not found at $ENV_FILE"
  echo "Copy .env.example to .env and fill in your SSH credentials."
  exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

# ── Validate required variables ────────────────────────────────────────
missing=()
[ -z "${SSH_KEY_PATH:-}" ]      && missing+=("SSH_KEY_PATH")
[ -z "${SSH_DIRECTORY:-}" ]     && missing+=("SSH_DIRECTORY")
[ -z "${DROIDS_IP_ADDRESS:-}" ] && missing+=("DROIDS_IP_ADDRESS")

if [ ${#missing[@]} -gt 0 ]; then
  echo "Error: Missing required environment variable(s): ${missing[*]}"
  exit 1
fi

# Expand ~ in SSH_KEY_PATH
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

# ── Validate SSH key ──────────────────────────────────────────────────
if [ "$DRY_RUN" = false ] && [ ! -f "$SSH_KEY_PATH" ]; then
  echo "Error: SSH key not found at $SSH_KEY_PATH"
  exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────
REMOTE_POLICY="$SSH_DIRECTORY/locomotion/outputs/policy.onnx"
LOCAL_POLICY="$SCRIPT_DIR/locomotion/outputs/policy.onnx"

REMOTE_VIDEO="$SSH_DIRECTORY/locomotion/outputs/videos/latest_video.mp4"
LOCAL_VIDEO="$SCRIPT_DIR/locomotion/sim-outputs/media/latest_video.mp4"

# ── Ensure local directories exist ─────────────────────────────────────
mkdir -p "$SCRIPT_DIR/locomotion/outputs"
if [ "$DOWNLOAD_VIDEO" = true ]; then
  mkdir -p "$SCRIPT_DIR/locomotion/sim-outputs/media"
fi

# ── Download policy ────────────────────────────────────────────────────
SCP_CMD="scp -i $SSH_KEY_PATH $DROIDS_IP_ADDRESS:$REMOTE_POLICY $LOCAL_POLICY"

if [ "$DRY_RUN" = true ]; then
  echo "[dry-run] $SCP_CMD"
else
  echo "Downloading policy..."
  $SCP_CMD
  echo "Policy saved to $LOCAL_POLICY"
fi

# ── Download video (optional) ─────────────────────────────────────────
if [ "$DOWNLOAD_VIDEO" = true ]; then
  SCP_VIDEO_CMD="scp -i $SSH_KEY_PATH $DROIDS_IP_ADDRESS:$REMOTE_VIDEO $LOCAL_VIDEO"

  if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] $SCP_VIDEO_CMD"
  else
    echo "Downloading video..."
    $SCP_VIDEO_CMD
    echo "Video saved to $LOCAL_VIDEO"
  fi
fi

# ── Launch teleop ─────────────────────────────────────────────────────
if [ "$DOWNLOAD_ONLY" = true ]; then
  echo "Download complete (--download-only). Skipping teleop."
  exit 0
fi

TELEOP_CMD="mjpython $SCRIPT_DIR/locomotion/teleop.py --policy $LOCAL_POLICY"

if [ "$DRY_RUN" = true ]; then
  echo "[dry-run] $TELEOP_CMD"
else
  echo "Launching teleop..."
  exec $TELEOP_CMD
fi
