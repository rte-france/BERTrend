#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import APIRouter

from bertrend.services.summary_server.models import SummarizerInfo
from bertrend.services.summary_server.routers.summarize import (
    SUMMARIZER_REGISTRY,
    _summarizer_cache,
)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/summarizers", response_model=list[SummarizerInfo])
def list_summarizers():
    return [
        SummarizerInfo(
            name=name,
            description=meta["description"],
            requires_api_key=meta["requires_api_key"],
            requires_gpu=meta["requires_gpu"],
            loaded=name in _summarizer_cache,
        )
        for name, meta in SUMMARIZER_REGISTRY.items()
    ]
