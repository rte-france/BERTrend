#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import sys

import requests

from bertrend.bertrend_apps.services.bertrend.config.settings import get_config

# Load config
CONFIG = get_config()

# Modify hostname if needed
if CONFIG.host == "0.0.0.0":
    CONFIG.host = "localhost"

# Try to check health endpoint and handle any connection errors
try:
    with requests.get(
        f"https://{CONFIG.host}:{CONFIG.port}/health", timeout=5, verify=False
    ) as response:
        if response.status_code == 200:
            sys.exit(0)
        else:
            sys.exit(1)

# If an error occurs, exit with status code 1
except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
    sys.exit(1)
