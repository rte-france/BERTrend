#
# Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#

# Create directory on host
mkdir -p ~/.bertrend/db

# Export your UID/GID (add to your shell profile for persistence)
export UID=$(id -u)
export GID=$(id -g)

# Run docker compose
docker compose up --build --force-recreate -d