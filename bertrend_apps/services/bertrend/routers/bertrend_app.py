#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend_apps.services.bertrend.models.bertrend_app_models import (
    GenerateReportRequest,
    RegenerateRequest,
    StatusResponse,
    TrainNewModelRequest,
)
from bertrend_apps.services.queue_management.queue_manager import QueueManager
from bertrend_apps.services.queue_management.rabbitmq_config import RabbitMQConfig

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
    try:
        config = RabbitMQConfig()
        queue_manager = QueueManager(config)

        request_data = {
            "endpoint": "/train-new-model",
            "method": "POST",
            "json_data": req.model_dump(),
        }

        correlation_id = await queue_manager.publish_request(request_data, priority=10)
        await queue_manager.close()

        return StatusResponse(
            status="queued",
            message=f"Train new model request for user '{req.user}' and model '{req.model_id}' queued successfully (correlation_id: {correlation_id})",
        )
    except Exception as e:
        logger.error(f"Error queuing train-new-model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        config = RabbitMQConfig()
        queue_manager = QueueManager(config)

        request_data = {
            "endpoint": "/regenerate",
            "method": "POST",
            "json_data": req.model_dump(),
        }

        correlation_id = await queue_manager.publish_request(request_data, priority=2)
        await queue_manager.close()

        return StatusResponse(
            status="queued",
            message=f"Regenerate request for user '{req.user}' and model '{req.model_id}' queued successfully (correlation_id: {correlation_id})",
        )
    except Exception as e:
        logger.error(f"Error queuing regenerate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        config = RabbitMQConfig()
        queue_manager = QueueManager(config)

        request_data = {
            "endpoint": "/generate-report",
            "method": "POST",
            "json_data": req.model_dump(),
        }

        correlation_id = await queue_manager.publish_request(request_data, priority=7)
        await queue_manager.close()

        return StatusResponse(
            status="queued",
            message=f"Generate report request for user '{req.user}' and model '{req.model_id}' queued successfully (correlation_id: {correlation_id})",
        )
    except Exception as e:
        logger.error(f"Error queuing generate-report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
