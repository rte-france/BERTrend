#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend_apps.prospective_demo.process_new_data import (
    regenerate_models,
    train_new_model,
)
from bertrend_apps.services.models.bertrend_app_models import (
    TrainNewModelRequest,
    TrainNewModelResponse,
    RegenerateRequest,
    RegenerateResponse,
)

router = APIRouter()


@router.post(
    "/train-new-model",
    response_model=TrainNewModelResponse,
    summary="Train new BERTrend model incrementally",
)
async def train_new(req: TrainNewModelRequest):
    """
    Incrementally enrich the BERTrend model with new data.

    This endpoint processes new data for a specific user and model ID,
    filtering data according to the configured granularity and training
    a new model for the most recent period.
    """
    try:
        result = await asyncio.to_thread(
            train_new_model,
            model_id=req.model_id,
            user_name=req.user_name,
        )
        return TrainNewModelResponse(**result)
    except Exception as e:
        logger.error(f"Error training new model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/regenerate",
    response_model=RegenerateResponse,
    summary="Regenerate models from scratch",
)
async def regenerate(req: RegenerateRequest):
    """
    Regenerate past models from scratch.

    This endpoint regenerates all models for a specific user and model ID,
    optionally with LLM-based analysis and filtering by a start date.
    """
    try:
        # Regenerate models
        await asyncio.to_thread(
            regenerate_models,
            model_id=req.model_id,
            user=req.user,
            with_analysis=req.with_analysis,
            since=pd.Timestamp(req.since) if req.since else None,
        )

        return RegenerateResponse(
            status="success",
            message=f"Successfully regenerated models for user '{req.user}' and model '{req.model_id}'",
        )

    except Exception as e:
        logger.error(f"Error regenerating models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
