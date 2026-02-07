#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os

from bertrend.bertrend_apps.services.bertrend.config.settings import get_config

# Load the configuration
CONFIG = get_config()

# Stop processes associated to API port
os.system(f"kill $(lsof -t -i:{CONFIG.port})")
