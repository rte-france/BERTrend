#!/bin/bash
#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Stop the lightweight BERTrend stack started by start_bertrend_lightweight.sh
# (see docker-compose.lightweight.yml). The external embedding server is not
# managed here: it runs elsewhere and is left untouched.
#
# Usage:
#   ./stop_bertrend_lightweight.sh              # stop and remove the containers
#   ./stop_bertrend_lightweight.sh --volumes    # also drop anonymous volumes
#   ./stop_bertrend_lightweight.sh --keep       # just stop, keep the containers

set -e

# Always operate from the repository root (where the compose file lives).
cd "$(dirname "$0")"

usage() {
    cat <<'USAGE'
Stop the lightweight BERTrend stack started by start_bertrend_lightweight.sh.
The external embedding server is not managed here and is left untouched.

Usage:
  ./stop_bertrend_lightweight.sh              stop and remove the containers
  ./stop_bertrend_lightweight.sh -v|--volumes also drop anonymous volumes
  ./stop_bertrend_lightweight.sh -k|--keep    just stop, keep the containers
  ./stop_bertrend_lightweight.sh -h|--help    show this message
USAGE
}

COMPOSE_FILE="docker-compose.lightweight.yml"

REMOVE_VOLUMES=false
KEEP_CONTAINERS=false

while [ $# -gt 0 ]; do
    case "$1" in
        -v|--volumes)
            REMOVE_VOLUMES=true
            ;;
        -k|--keep)
            KEEP_CONTAINERS=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option '$1' (try --help)" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

# docker-compose.lightweight.yml marks EMBEDDING_SERVICE_URL and
# BERTREND_CLIENT_SECRET as required (${VAR:?...}). Compose interpolates the
# file for *every* command, so `down` fails too when they are unset. Neither
# value matters for teardown, so supply placeholders when they are absent.
export EMBEDDING_SERVICE_URL=${EMBEDDING_SERVICE_URL:-unused-when-stopping}
export BERTREND_CLIENT_SECRET=${BERTREND_CLIENT_SECRET:-unused-when-stopping}

# Referenced by the volume definitions; defaults mirror the compose file so the
# paths resolve to the same project compose would otherwise compute.
export HF_HOME=${HF_HOME:-"${HOME}/.cache/huggingface"}
export BERTREND_BASE_DIR=${BERTREND_BASE_DIR:-".bertrend"}

if [ "$KEEP_CONTAINERS" = true ]; then
    echo "Stopping the lightweight BERTrend stack (containers kept)..."
    docker compose -f "$COMPOSE_FILE" stop
else
    echo "Stopping and removing the lightweight BERTrend stack..."
    if [ "$REMOVE_VOLUMES" = true ]; then
        # NB: the stack declares no named volumes, so this only drops anonymous
        # ones (e.g. RabbitMQ state). Bind mounts such as BERTREND_BASE_DIR and
        # HF_HOME live on the host and are never removed.
        docker compose -f "$COMPOSE_FILE" down --volumes
    else
        docker compose -f "$COMPOSE_FILE" down
    fi
fi

echo "Done. Remaining containers for this stack:"
docker compose -f "$COMPOSE_FILE" ps --all
