#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import uvicorn
from fastapi import FastAPI

from bertrend.services.scheduling.routers import scheduling, info
from bertrend.services.scheduling.routers.scheduling import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(info.router, tags=["Info"])
app.include_router(scheduling.router, tags=["Scheduler"])

if __name__ == "__main__":
    uvicorn.run(
        "bertrend.services.scheduling.scheduling_service:app",
        host="0.0.0.0",
        port=8882,
        reload=True,
    )
