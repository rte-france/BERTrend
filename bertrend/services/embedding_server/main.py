#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import FastAPI

from bertrend.services.embedding_server.routers import authentication, embeddings, info

app = FastAPI()
app.include_router(info.router, tags=["Info"])
app.include_router(embeddings.router, tags=["Embedding"])
app.include_router(authentication.router, tags=["Authentication"])
