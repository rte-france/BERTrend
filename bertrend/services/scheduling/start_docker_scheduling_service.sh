#!/bin/bash
#
# Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#
# Start the standalone APScheduler service (see docker-compose.yml next to this
# script). HOST_UID/HOST_GID are resolved here rather than in the compose file
# because they depend on the host: they make the container write the SQLite job
# store as the invoking user instead of root.

set -e

# Always operate from the directory holding the compose file.
cd "$(dirname "$0")"

# Create the host directory backing the mounted job store.
mkdir -p ~/.bertrend/db

# User/group IDs so files written into the mounted volume are owned by the host
# user. NB: do not use UID/GID -- UID is readonly in bash, so `export UID=...`
# fails and compose silently falls back to 1000.
HOST_UID=$(id -u)
HOST_GID=$(id -g)
export HOST_UID HOST_GID

echo "Starting the BERTrend scheduling service with:"
echo "  HOST_UID=$HOST_UID"
echo "  HOST_GID=$HOST_GID"

docker compose up --build --force-recreate -d

echo "Scheduler started: http://localhost:8882"
