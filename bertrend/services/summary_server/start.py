#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import uvicorn

from bertrend.services.summary_server.config.settings import get_config

CONFIG = get_config()

if __name__ == "__main__":
    uvicorn.run(
        "bertrend.services.summary_server.main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        workers=CONFIG.number_workers,
    )
