#!/bin/bash

#Author: The GOAT Oren Gershony

#############################################################################
# Connects to SSH server via port forwarding and load latest training video #    
#############################################################################

set -e

# Pull local variables from .env file
source .env

echo "Viewing latest training results..."

#1. SSH into node and host a remote python server
echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

ssh -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" << EOF
  set -e

  echo "Connected to SSH node"

  cd ~/$SSH_DIRECTORY/pupper-simulations/locomotion

  # Kill any existing HTTP server on this port
  echo "Checking for existing HTTP server on port $SSH_PORT..."
  lsof -ti:$SSH_PORT | xargs kill -9 2>/dev/null || echo "No existing server found"

  # Start a simple HTTP server at port 8000
  uv run python -m http.server $SSH_PORT
EOF

#2. Setup port forwarding to local machine
echo "Setting up port forwarding from remote server to local machine..."

ssh -L $SSH_PORT:localhost:$SSH_PORT -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS"

echo "Port forwarding established. You can now view the training results by opening your web browser and navigating to http://localhost:$SSH_PORT"





