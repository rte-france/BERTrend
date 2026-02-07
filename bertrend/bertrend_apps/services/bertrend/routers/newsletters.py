#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend.bertrend_apps import SCHEDULER_UTILS
from bertrend.bertrend_apps.services.bertrend.config.settings import get_config
from bertrend.bertrend_apps.services.bertrend.models.newsletters_models import (
    NewsletterRequest,
)
from bertrend.bertrend_apps.services.queue_management.queue_manager import QueueManager
from bertrend.bertrend_apps.services.queue_management.rabbitmq_config import (
    RabbitMQConfig,
)

# Load the configuration
CONFIG = get_config()

router = APIRouter()


@router.post(
    "/generate-newsletters",
    summary="Generate newsletter from feed",
)
async def newsletter_from_feed(req: NewsletterRequest):
    """
    Creates a newsletter associated to a data feed.
    """
    try:
        config = RabbitMQConfig()
        queue_manager = QueueManager(config)

        request_data = {
            "endpoint": "/generate-newsletters",
            "method": "POST",
            "json_data": req.model_dump(),
        }

        correlation_id = await queue_manager.publish_request(request_data, priority=5)
        await queue_manager.close()

        return {
            "status": "queued",
            "message": "Newsletter generation request queued successfully",
            "correlation_id": correlation_id,
        }
    except Exception as e:
        logger.error(f"Error queuing newsletter generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule-newsletters", summary="Schedule newsletter automation")
async def automate_newsletter(req: NewsletterRequest):
    """
    Schedule data scrapping on the basis of a feed configuration file.
    """
    try:
        await asyncio.to_thread(
            SCHEDULER_UTILS.schedule_newsletter,
            req.newsletter_toml_path,
            req.data_feed_toml_path,
        )
        return {"status": "Newsletter scheduling completed successfully"}
    except Exception as e:
        logger.error(f"Error scheduling newsletter: {e}")
        raise HTTPException(status_code=500, detail=str(e))
