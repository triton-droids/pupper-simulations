#!/usr/bin/env bash
set -euo pipefail

# Author: The GOAT Oren Gershony (with Windows fixes)

TEST_MODE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) TEST_MODE=true; shift ;;
    *) echo "Unknown option: $1"; echo "Usage: $0 [--test]"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

SSH_PORT="${SSH_PORT:-22}"          # SSH port on the server (usually 22)
HTTP_PORT="${HTTP_PORT:-8123}"      # Local-forwarded HTTP port for viewing results

# Safety: never run the HTTP server on the SSH port or a privileged port
if [[ "$HTTP_PORT" == "$SSH_PORT" || "$HTTP_PORT" -lt 1024 ]]; then
  echo "ERROR: HTTP_PORT=$HTTP_PORT is unsafe. Pick a non-privileged port not equal to SSH_PORT (example 8123)."
  exit 1
fi

if [[ "$TEST_MODE" == true ]]; then
  OUTPUT_DIR="outputs/bittle_test_latest"
  echo "Viewing TEST results..."
else
  OUTPUT_DIR="outputs/bittle_train_latest"
  echo "Viewing latest training results..."
fi

echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

REMOTE_PIDFILE="/tmp/pupper_http_${HTTP_PORT}.pid"
REMOTE_LOG="/tmp/pupper_http_${HTTP_PORT}.log"

# 1) Start (or restart) a simple HTTP server on the remote box
ssh -p "$SSH_PORT" -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" bash -s -- \
  "$SSH_DIRECTORY" "$HTTP_PORT" "$REMOTE_PIDFILE" "$REMOTE_LOG" <<'REMOTE'
set -euo pipefail
REPO_DIR="$1"
PORT="$2"
PIDFILE="$3"
LOGFILE="$4"

echo "Connected to SSH node"

# Try both possible layouts (your repo path has differed across scripts)
if [[ -d "$REPO_DIR/locomotion" ]]; then
  cd "$REPO_DIR/locomotion"
elif [[ -d "$REPO_DIR/pupper-simulations/locomotion" ]]; then
  cd "$REPO_DIR/pupper-simulations/locomotion"
else
  echo "ERROR: Could not find locomotion directory under: $REPO_DIR"
  exit 1
fi

# Kill previous server using PID file if present
if [[ -f "$PIDFILE" ]]; then
  oldpid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${oldpid:-}" ]] && kill -0 "$oldpid" 2>/dev/null; then
    kill "$oldpid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

# If fuser exists, also try to free the port (best effort)
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi

# Start HTTP server in background and record PID
nohup uv run python -m http.server "$PORT" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "HTTP server started on port $PORT (PID: $(cat "$PIDFILE"))"
REMOTE

# 2) Start SSH port forwarding locally, detached, and record PID
mkdir -p "$SCRIPT_DIR/locomotion/sim-outputs/policies" "$SCRIPT_DIR/locomotion/sim-outputs/media"
LOCAL_TUNNEL_PIDFILE="$SCRIPT_DIR/locomotion/sim-outputs/.tunnel_${HTTP_PORT}.pid"

echo "Setting up port forwarding localhost:$HTTP_PORT -> remote localhost:$HTTP_PORT ..."
nohup ssh -L "${HTTP_PORT}:localhost:${HTTP_PORT}" -p "$SSH_PORT" -i "$SSH_KEY_PATH" \
  -o ExitOnForwardFailure=yes \
  "$DROIDS_IP_ADDRESS" -N > /tmp/pupper_tunnel_${HTTP_PORT}.log 2>&1 &

echo $! > "$LOCAL_TUNNEL_PIDFILE"
sleep 2
echo "Port forwarding established (local PID: $(cat "$LOCAL_TUNNEL_PIDFILE"))"

echo "Downloading results..."
curl -fL -o "$SCRIPT_DIR/locomotion/sim-outputs/media/latest_video.mp4" \
  "http://localhost:${HTTP_PORT}/${OUTPUT_DIR}/videos/latest_video.mp4"

curl -fL -o "$SCRIPT_DIR/locomotion/sim-outputs/policies/policy.onnx" \
  "http://localhost:${HTTP_PORT}/${OUTPUT_DIR}/policy.onnx"

VIDEO_PATH="$SCRIPT_DIR/locomotion/sim-outputs/media/latest_video.mp4"

# 3) Reveal the video in the file explorer (Windows/macOS/Linux)
uname_out="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$uname_out" in
  darwin*) open -R "$VIDEO_PATH" ;;
  mingw*|msys*|cygwin*)
    if command -v cygpath >/dev/null 2>&1; then
      WINPATH="$(cygpath -w "$VIDEO_PATH")"
      explorer.exe /select,"$WINPATH" >/dev/null 2>&1 || true
    else
      explorer.exe /select,"$VIDEO_PATH" >/dev/null 2>&1 || true
    fi
    ;;
  *)
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$(dirname "$VIDEO_PATH")" >/dev/null 2>&1 || true
    fi
    ;;
esac

echo ""
echo "Results downloaded successfully!"
echo "  Policy: locomotion/sim-outputs/policies/policy.onnx"
echo "  Video:  locomotion/sim-outputs/media/latest_video.mp4"
echo ""
echo "To stop the local port forwarding tunnel:"
echo "  kill $(cat "$LOCAL_TUNNEL_PIDFILE")"
echo ""
echo "To stop the remote HTTP server:"
echo "  ssh -p $SSH_PORT -i $SSH_KEY_PATH $DROIDS_IP_ADDRESS \"kill \$(cat $REMOTE_PIDFILE) 2>/dev/null || true\""