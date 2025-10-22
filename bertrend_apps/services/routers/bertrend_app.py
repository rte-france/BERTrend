#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio
from datetime import timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend.utils.data_loading import split_data
from bertrend_apps.prospective_demo.process_new_data import (
    load_all_data,
    get_model_config,
    train_new_model_for_period,
    regenerate_models,
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
async def train_new_model(req: TrainNewModelRequest):
    """
    Incrementally enrich the BERTrend model with new data.

    This endpoint processes new data for a specific user and model ID,
    filtering data according to the configured granularity and training
    a new model for the most recent period.
    """
    try:
        logger.info(
            f'Processing new data for user "{req.user_name}" about "{req.model_id}"...'
        )

        # Get relevant model info from config
        model_config = await asyncio.to_thread(
            get_model_config, model_id=req.model_id, user=req.user_name
        )
        granularity = model_config["granularity"]
        language_code = model_config["language"]
        split_by_paragraph = model_config.get("split_by_paragraph", True)

        logger.info(f"Splitting data by paragraphs: {split_by_paragraph}")

        # Load data for last period
        new_data = await asyncio.to_thread(
            load_all_data,
            model_id=req.model_id,
            user=req.user_name,
            language_code=language_code,
        )

        if new_data is None or new_data.empty:
            return TrainNewModelResponse(
                status="no_data",
                message=f"No new data found for model '{req.model_id}'",
            )

        # Filter data according to granularity
        # Calculate the date X days ago
        reference_timestamp = pd.Timestamp(
            new_data["timestamp"].max().date()
        )  # used to identify the last model
        cut_off_date = new_data["timestamp"].max() - timedelta(days=granularity)
        # Filter the DataFrame to keep only the rows within the last X days
        filtered_df = new_data[new_data["timestamp"] >= cut_off_date]

        # Split data by paragraphs
        filtered_df = await asyncio.to_thread(
            split_data,
            df=filtered_df,
            split_by_paragraph="yes" if split_by_paragraph else "no",
        )

        # Train new model for this period
        await asyncio.to_thread(
            train_new_model_for_period,
            model_id=req.model_id,
            user_name=req.user_name,
            new_data=filtered_df,
            reference_timestamp=reference_timestamp,
        )

        return TrainNewModelResponse(
            status="success",
            message=f"Successfully trained new model for user '{req.user_name}' and model '{req.model_id}'",
        )

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
        logger.info(
            f"Regenerating models for user '{req.user}' about '{req.model_id}', "
            f"with analysis: {req.with_analysis}..."
        )

        # Convert string date to pd.Timestamp if provided
        since_timestamp = pd.Timestamp(req.since) if req.since else None

        # Regenerate models
        await asyncio.to_thread(
            regenerate_models,
            model_id=req.model_id,
            user=req.user,
            with_analysis=req.with_analysis,
            since=since_timestamp,
        )

        return RegenerateResponse(
            status="success",
            message=f"Successfully regenerated models for user '{req.user}' and model '{req.model_id}'",
        )

    except Exception as e:
        logger.error(f"Error regenerating models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
