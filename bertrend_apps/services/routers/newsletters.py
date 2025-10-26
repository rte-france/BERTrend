#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio
import os

from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend import BEST_CUDA_DEVICE

from bertrend_apps import SCHEDULER_UTILS
from bertrend_apps.newsletters.newsletter_generation import process_newsletter

from bertrend_apps.services.config.settings import get_config
from bertrend_apps.services.models.newsletters_models import (
    NewsletterRequest,
    NewsletterResponse,
    ScheduleNewsletterRequest,
)

# Load the configuration
CONFIG = get_config()

router = APIRouter()


@router.post(
    "/newsletters",
    response_model=NewsletterResponse,
    summary="Generate newsletter from feed",
)
async def newsletter_from_feed(req: NewsletterRequest):
    """
    Creates a newsletter associated to a data feed.
    """
    try:
        await asyncio.to_thread(
            process_newsletter,
            req.newsletter_toml_path,
            req.data_feed_toml_path,
        )
        return {"status": "Newsletter generated successfully"}
    except Exception as e:
        logger.error(f"Error generating newsletter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule-newsletters", summary="Schedule newsletter automation")
async def automate_newsletter(req: ScheduleNewsletterRequest):
    """
    Schedule data scrapping on the basis of a feed configuration file.
    """
    try:
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        await asyncio.to_thread(
            SCHEDULER_UTILS.schedule_newsletter,
            req.newsletter_toml_cfg_path,
            req.data_feed_toml_cfg_path,
            cuda_devices,
        )
        return {"status": "Newsletter scheduling completed successfully"}
    except Exception as e:
        logger.error(f"Error scheduling newsletter: {e}")
        raise HTTPException(status_code=500, detail=str(e))
