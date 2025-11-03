#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/", summary="Redirect to API documentation")
def root():
    """Redirect root path to /docs"""
    return RedirectResponse(url="/docs")
