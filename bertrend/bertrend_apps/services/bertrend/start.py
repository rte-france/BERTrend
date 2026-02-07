#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import uvicorn

from bertrend.bertrend_apps.services.bertrend.config.settings import get_config

# Load the configuration first - NB. This will also load/set the CUDA_AVAILABLE_DEVICES from the .env

CONFIG = get_config()

# Start the FastAPI application
if __name__ == "__main__":
    uvicorn.run(
        "bertrend.bertrend_apps.services.bertrend.bertrend_service:app",
        host=CONFIG.host,
        port=CONFIG.port,
        workers=CONFIG.number_workers,
    )
