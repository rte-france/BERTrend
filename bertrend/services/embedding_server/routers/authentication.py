#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from typing import Annotated

from fastapi import Depends, APIRouter, Security
from fastapi.security import OAuth2PasswordRequestForm

from bertrend.services.embedding_server.security import (
    ADMIN,
    Token,
    TokenData,
    get_token,
    list_registered_clients,
    get_current_client,
    view_rate_limits,
)

router = APIRouter()


@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    return get_token(form_data)


@router.get(
    "/list_registered_clients", summary="List registered clients (requires admin scope)"
)
async def list_clients(
    current_client: TokenData = Security(get_current_client, scopes=[ADMIN]),
):
    return list_registered_clients()


@router.get(
    "/rate-limits", summary="Check current rate limit usage (requires admin scope)"
)
async def get_rate_limits(
    current_client: TokenData = Security(get_current_client, scopes=[ADMIN]),
):
    return view_rate_limits()
