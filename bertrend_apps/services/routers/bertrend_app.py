#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend_apps.prospective_demo.automated_report_generation import (
    generate_automated_report,
)
from bertrend_apps.prospective_demo.process_new_data import (
    regenerate_models,
    train_new_model,
)
from bertrend_apps.services.utils.logging_utils import get_file_logger
from bertrend_apps.services.models.bertrend_app_models import (
    TrainNewModelRequest,
    RegenerateRequest,
    StatusResponse,
    GenerateReportRequest,
)

router = APIRouter()


@router.post(
    "/train-new-model",
    response_model=StatusResponse,
    summary="Train new BERTrend model incrementally",
)
async def train_new(req: TrainNewModelRequest):
    """
    Incrementally enrich the BERTrend model with new data.

    This endpoint processes new data for a specific user and model ID,
    filtering data according to the configured granularity and training
    a new model for the most recent period.
    """
    # Create a unique log file for this call
    logger_id = get_file_logger(
        id="train-new-model", user_name=req.user, model_id=req.model_id
    )

    try:
        result = await asyncio.to_thread(
            train_new_model,
            model_id=req.model_id,
            user_name=req.user,
        )
        return StatusResponse(**result)
    except Exception as e:
        logger.error(f"Error training new model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove the logger to prevent writing to this file for other calls
        logger.remove(logger_id)


@router.post(
    "/regenerate",
    response_model=StatusResponse,
    summary="Regenerate models from scratch",
)
async def regenerate(req: RegenerateRequest):
    """
    Regenerate past models from scratch.

    This endpoint regenerates all models for a specific user and model ID,
    optionally with LLM-based analysis and filtering by a start date.
    """
    # Create a unique log file for this call
    logger_id = get_file_logger(
        id="regenerate", user_name=req.user, model_id=req.model_id
    )

    try:
        # Regenerate models
        await asyncio.to_thread(
            regenerate_models,
            model_id=req.model_id,
            user=req.user,
            with_analysis=req.with_analysis,
            since=pd.Timestamp(req.since) if req.since else None,
        )

        return StatusResponse(
            status="success",
            message=f"Successfully regenerated models for user '{req.user}' and model '{req.model_id}'",
        )

    except Exception as e:
        logger.error(f"Error regenerating models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove the logger to prevent writing to this file for other calls
        logger.remove(logger_id)


@router.post(
    "/generate-report",
    response_model=StatusResponse,
    summary="Generate and send an automated report",
)
async def generate_report(req: GenerateReportRequest):
    """
    Generate and send an automated report based on model configuration.

    This endpoint generates a report for a specific user and model ID,
    optionally using a reference date. If no reference date is provided,
    it uses the most recent data available.
    """
    # Create a unique log file for this call
    logger_id = get_file_logger(
        id="regenerate", user_name=req.user, model_id=req.model_id
    )

    try:
        # Generate report
        await asyncio.to_thread(
            generate_automated_report,
            user=req.user,
            model_id=req.model_id,
            reference_date=req.reference_date,
        )

        return StatusResponse(
            status="success",
            message=f"Successfully generated report for user '{req.user}' and model '{req.model_id}'",
        )

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Remove the logger to prevent writing to this file for other calls
        logger.remove(logger_id)
