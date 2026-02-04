#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend_apps import SCHEDULER_UTILS
from bertrend_apps.data_provider.data_provider_utils import (
    generate_query_file,
)
from bertrend_apps.services.bertrend.config.settings import get_config
from bertrend_apps.services.bertrend.models.data_provider_models import (
    AutoScrapeRequest,
    GenerateQueryFileRequest,
    GenerateQueryFileResponse,
    ScrapeFeedRequest,
    ScrapeRequest,
    ScrapeResponse,
)
from bertrend_apps.services.queue_management.queue_manager import QueueManager
from bertrend_apps.services.queue_management.rabbitmq_config import RabbitMQConfig

# Load the configuration
CONFIG = get_config()

# FastAPI application
router = APIRouter()


# Endpoints
@router.post(
    "/scrape",
    response_model=ScrapeResponse,
    summary="Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing or NewsCatcher news (single request).",
)
async def scrape_api(req: ScrapeRequest):
    try:
        config = RabbitMQConfig()
        queue_manager = QueueManager(config)

        request_data = {
            "endpoint": "/scrape",
            "method": "POST",
            "json_data": req.model_dump(),
        }

        correlation_id = await queue_manager.publish_request(
            request_data,
            priority=2,
        )
        await queue_manager.close()

        return ScrapeResponse(
            stored_path=req.save_path.resolve() if req.save_path else None,
            article_count=0,  # Not known yet
            status="queued",
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(f"Error queuing scrape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/auto-scrape",
    response_model=ScrapeResponse,
    summary="Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher (multiple requests from a configuration file: each line of the file shall be compliant with the following format:        <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)",
)
async def auto_scrape_api(req: AutoScrapeRequest):
    try:
        config = RabbitMQConfig()
        queue_manager = QueueManager(config)

        request_data = {
            "endpoint": "/auto-scrape",
            "method": "POST",
            "json_data": req.model_dump(),
        }

        correlation_id = await queue_manager.publish_request(
            request_data,
            priority=2,
        )
        await queue_manager.close()

        return ScrapeResponse(
            stored_path=req.save_path.resolve() if req.save_path else None,
            article_count=0,  # Not known yet
            status="queued",
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(f"Error queuing auto-scrape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generate-query-file",
    response_model=GenerateQueryFileResponse,
    summary="Generates a query file to be used with the auto-scrape command. This is useful for queries generating many results."
    "This will split the broad query into many ones, each one covering an 'interval' (range) in days covered by each atomic"
    "request. If you want to cover several keywords, run the command several times with the same output file.",
)
async def generate_query_file_api(req: GenerateQueryFileRequest):
    try:
        line_count = await asyncio.to_thread(
            generate_query_file,
            req.keywords,
            req.after,
            req.before,
            req.interval,
            req.save_path,
        )
        return GenerateQueryFileResponse(
            save_path=req.save_path.resolve(), line_count=line_count
        )
    except Exception as e:
        logger.error(f"Error scrapping data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/scrape-feed",
    response_model=ScrapeResponse,
    summary="Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher on the basis of a feed configuration file",
)
async def scrape_from_feed_api(req: ScrapeFeedRequest):
    """
    Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher on the basis of feed configuration.

    This is the async equivalent of the CLI scrape-feed command.
    """
    try:
        config = RabbitMQConfig()
        queue_manager = QueueManager(config)

        request_data = {
            "endpoint": "/scrape-feed",
            "method": "POST",
            "json_data": req.model_dump(),
        }

        correlation_id = await queue_manager.publish_request(
            request_data,
            priority=6,
        )
        await queue_manager.close()

        return ScrapeResponse(
            stored_path=None,  # Not known yet
            article_count=0,  # Not known yet
            status="queued",
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(f"Error queuing scrape-feed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error queuing scrape-feed: {str(e)}"
        )


@router.post(
    "/schedule-scrapping",
    summary="Schedule data scrapping on the basis of a feed configuration file",
)
async def automate_scrapping_api(req: ScrapeFeedRequest):
    try:
        await asyncio.to_thread(SCHEDULER_UTILS.schedule_scrapping, Path(req.feed_cfg))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "scheduled"}
