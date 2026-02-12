#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import FastAPI

from bertrend.services.summary_server.routers import info, summarize

app = FastAPI(title="BERTrend Summarization Service")
app.include_router(info.router, tags=["Info"])
app.include_router(summarize.router, tags=["Summarization"])
