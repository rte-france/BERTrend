#!/bin/bash

# Use it like docker-compose
#./run_compose_script.sh up -d
#./run_compose_script.sh up  --build --no-deps --force-recreate -d
#./run_compose_script.sh down
#./run_compose_script.sh logs -f

# Set user ID and group ID for proper file permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Set HF_HOME if not already set (defaults to ~/.cache/huggingface)
export HF_HOME=${HF_HOME:-"${HOME}/.cache/huggingface"}

# Create the HF_HOME directory if it doesn't exist
mkdir -p "$HF_HOME"

echo "Starting docker-compose with:"
echo "  HOST_UID: $HOST_UID"
echo "  HOST_GID: $HOST_GID"
echo "  HF_HOME: $HF_HOME"

# Run docker-compose with the environment variables
docker compose "$@"