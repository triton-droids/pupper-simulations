#!/bin/bash

#Author: The GOAT Oren Gershony

############################################
# Deployment script for remote training    #
############################################

set -e

# Pull local variables from .env file
source .env


echo "Starting deployment process..."

# 1. Stage, commit, and push current working changes
echo "Staging and comitting changes"

git add .

git commit -m "Deploying latest changes for training"

git push -u origin $BRANCH_NAME

#2. SSH into the remote server and run training script

# SSH into node
echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

sshpass -p "$SSH_PASSWORD" ssh -i "$SSH_KEY_PATH" "$DROIDS_IP_ADDRESS" "bash -c '
  set -e
  cd ~/$SSH_DIRECTORY/pupper-simulations/locomotion
  git pull
  uv run train.py $@
'"



