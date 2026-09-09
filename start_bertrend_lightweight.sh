#!/bin/bash
#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Start the lightweight BERTrend stack: only the main application, with the
# embedding server expected to run elsewhere (see docker-compose.lightweight.yml).
#
# HOST_UID / HOST_GID are exported so files the container writes into the mounted
# volumes are owned by the host user (avoids root-owned files on the host).
#
# EMBEDDING_SERVICE_URL must point to your externally-running embedding server.
# Set it in the environment or in a .env file next to this script, e.g.:
#   EMBEDDING_SERVICE_URL=https://your-embedding-host:6464
#
# Usage:
#   ./start_bertrend_lightweight.sh

set -e

# Always operate from the repository root (where the compose file lives).
cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.lightweight.yml"

# Optionally activate a local Python virtualenv if present (harmless for Docker).
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Get the latest code (non-fatal if it fails, e.g. dirty tree / no upstream).
git pull || echo "warning: 'git pull' skipped/failed; continuing with current checkout"

# User/group IDs so mounted-volume files are owned by the host user.
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Locations mounted into the container (defaults mirror docker-compose.lightweight.yml).
export HF_HOME=${HF_HOME:-"${HOME}/.cache/huggingface"}
export BERTREND_BASE_DIR=${BERTREND_BASE_DIR:-".bertrend"}

echo "Using:"
echo "  HOST_UID=$HOST_UID"
echo "  HOST_GID=$HOST_GID"
echo "  HF_HOME=$HF_HOME"
echo "  BERTREND_BASE_DIR=$BERTREND_BASE_DIR"

# The lightweight compose requires an external embedding server URL.
if [ -z "${EMBEDDING_SERVICE_URL:-}" ] && ! grep -qE '^[[:space:]]*EMBEDDING_SERVICE_URL=' .env 2>/dev/null; then
    echo "error: EMBEDDING_SERVICE_URL is not set." >&2
    echo "       Export it or add it to a .env file, pointing to your embedding server:" >&2
    echo "       EMBEDDING_SERVICE_URL=https://your-embedding-host:6464" >&2
    exit 1
fi

# The external embedding server also requires client credentials (compose treats
# BERTREND_CLIENT_SECRET as mandatory; an empty value only fails later as a 401).
if [ -z "${BERTREND_CLIENT_SECRET:-}" ] && ! grep -qE '^[[:space:]]*BERTREND_CLIENT_SECRET=' .env 2>/dev/null; then
    echo "error: BERTREND_CLIENT_SECRET is not set." >&2
    echo "       Export it or add it to a .env file; it must match the client secret" >&2
    echo "       registered for the 'bertrend' client on your embedding server." >&2
    exit 1
fi

# APScheduler job store. The scheduler resolves it as Path.home()/.bertrend/db,
# so this is the invoking user's home -- the same file the service used before it
# was containerised -- and not something under BERTREND_BASE_DIR.
export SCHEDULER_DB_DIR=${SCHEDULER_DB_DIR:-"${HOME}/.bertrend/db"}

# Create the mounted host directories so they are owned by the current user.
# SCHEDULER_DB_DIR matters in particular: it is a separate bind-mount target, so
# if it does not exist Docker creates it as root and the scheduler -- which runs
# as HOST_UID -- cannot open its SQLite job store ("unable to open database
# file"), fails its healthcheck, and blocks the bertrend service depending on it.
mkdir -p "$BERTREND_BASE_DIR" "$SCHEDULER_DB_DIR" "$HF_HOME"

# A leftover root-owned job store from an earlier run cannot be fixed by
# mkdir -p; flag it rather than failing later with an opaque SQLite error.
if [ ! -w "$SCHEDULER_DB_DIR" ]; then
    echo "error: $SCHEDULER_DB_DIR is not writable by $(id -un) (uid $HOST_UID)." >&2
    echo "       It was probably created by Docker as root during a previous failed run." >&2
    echo "       Fix it with:" >&2
    echo "         sudo chown -R $HOST_UID:$HOST_GID $SCHEDULER_DB_DIR" >&2
    exit 1
fi

# (Re)build and start the stack.
docker compose -f "$COMPOSE_FILE" down
docker compose -f "$COMPOSE_FILE" up --build -d

echo "BERTrend (lightweight) started. Demos:"
echo "  Topic Analysis: http://localhost:8083"
echo "  Weak Signals:   http://localhost:8084"
echo "  Prospective:    http://localhost:8081"
