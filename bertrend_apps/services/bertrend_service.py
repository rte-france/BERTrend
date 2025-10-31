#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from fastapi import FastAPI

from bertrend_apps.services.routers import (
    info,
    data_provider,
    newsletters,
    bertrend_app,
)

app = FastAPI()

app.include_router(info.router, tags=["Info"])
app.include_router(data_provider.router, tags=["Data Provider"])
app.include_router(newsletters.router, tags=["Newsletters"])
app.include_router(bertrend_app.router, tags=["BERTrend application"])
