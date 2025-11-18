#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend import load_toml_config
from bertrend_apps import SCHEDULER_UTILS
from bertrend_apps.newsletters.newsletter_generation import (
    process_newsletter,
    NEWSLETTER_SECTION,
)

from bertrend_apps.services.config.settings import get_config
from bertrend_apps.services.models.newsletters_models import NewsletterRequest
from bertrend_apps.services.utils.logging_utils import get_file_logger

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
    config = load_toml_config(req.newsletter_toml_path)
    model_id = config.get(NEWSLETTER_SECTION).get("id")
    # Create a unique log file for this call
    logger_id = get_file_logger(
        id="generate_newsletters", user_name="", model_id=model_id
    )
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
    finally:
        # Remove the logger to prevent writing to this file for other calls
        logger.remove(logger_id)


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
