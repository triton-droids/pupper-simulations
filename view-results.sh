#!/bin/bash

#Author: The GOAT Oren Gershony

#############################################################################
# Connects to SSH server via port forwarding and load latest training video #
#############################################################################

# Parse command line arguments
TEST_MODE=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --test)
      TEST_MODE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--test]"
      exit 1
      ;;
  esac
done

set -e

# Pull local variables from .env file
source .env

# Determine output directory based on mode
if [ "$TEST_MODE" = true ]; then
  OUTPUT_DIR="outputs/bittle_test_latest"
  echo "Viewing TEST results..."
else
  OUTPUT_DIR="outputs/bittle_train_latest"
  echo "Viewing latest training results..."
fi

#1. SSH into node and host a remote python server
echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

sshpass -p "$SSH_PASSWORD" ssh -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" << EOF
  set -e

  echo "Connected to SSH node"

  cd ~/$SSH_DIRECTORY/pupper-simulations/locomotion

  # Kill any existing HTTP server on this port
  echo "Checking for existing processes on port $SSH_PORT..."

  # Try multiple methods to kill the process
  fuser -k $SSH_PORT/tcp 2>/dev/null && echo "Killed process using fuser" || echo "fuser found no process"

  # Wait for port to be released
  sleep 2

  # Verify port is available
  if ss -tulpn | grep ":$SSH_PORT " > /dev/null 2>&1; then
    echo "ERROR: Port $SSH_PORT is still in use:"
    ss -tulpn | grep ":$SSH_PORT "
    exit 1
  fi

  echo "Port $SSH_PORT is now available"

  # Start HTTP server in the background
  nohup uv run python -m http.server $SSH_PORT > /tmp/http_server_$SSH_PORT.log 2>&1 &

  echo "HTTP server started on port $SSH_PORT (PID: \$!)"
  sleep 1
EOF

#2. Setup port forwarding to local machine
echo "Setting up port forwarding from remote server to local machine..."

# Start SSH port forwarding in the background (-f flag)
sshpass -p "$SSH_PASSWORD" ssh -L $SSH_PORT:localhost:$SSH_PORT -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" -N -f

# Wait for the tunnel to establish
sleep 2

echo "Port forwarding established."

# Create local directories for organized storage
mkdir -p locomotion/sim-outputs/policies
mkdir -p locomotion/sim-outputs/media

echo "Downloading results..."

# Download latest video
curl -L -o locomotion/sim-outputs/media/latest_video.mp4 "http://localhost:$SSH_PORT/$OUTPUT_DIR/videos/latest_video.mp4"

# Download trained policy (ONNX format)
curl -L -o locomotion/sim-outputs/policies/policy.onnx "http://localhost:$SSH_PORT/$OUTPUT_DIR/policy.onnx"

# Open video in file system viewer
open -R locomotion/sim-outputs/media/latest_video.mp4

echo ""
echo "Results downloaded successfully!"
echo "  Policy: locomotion/sim-outputs/policies/policy.onnx"
echo "  Video: locomotion/sim-outputs/media/latest_video.mp4"
echo ""
echo "Video location revealed in Finder. Port forwarding is running in the background."
echo "To stop port forwarding later, run: pkill -f 'ssh.*$SSH_PORT:localhost:$SSH_PORT'"





