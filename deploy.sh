#!/bin/bash

############################################
# Deployment script for remote training    #
############################################

# Pull local variables from .env file
source .env

# Configuration
DROIDS_IP_ADDRESS="tritondroids@132.249.64.152"

echo "Starting deployment process..."

# 1. Stage, commit, and push current working changes
echo "Staging and comitting changes"

git add .

git commit -m "Deploying latest changes for training"

git push -u origin $BRANCH_NAME

#2. SSH into the remote server

# SSH into node
echo "Connecting to remote server at $DROIDS_IP_ADDRESS"

ssh -i $SSH_KEY_PATH $DROIDS_IP_ADDRESS

cd $SSH_DIRECTORY/pupper-simulations/locomotion

# Pull changes from branch
git pull

# Run training script
uv run train.py




