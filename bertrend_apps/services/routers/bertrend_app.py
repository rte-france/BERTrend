#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from fastapi import APIRouter

from bertrend_apps.services.config.settings import get_config

# Load the configuration
CONFIG = get_config()

router = APIRouter()
