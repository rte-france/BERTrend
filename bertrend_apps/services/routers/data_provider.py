#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

from fastapi import APIRouter

import asyncio

from fastapi import HTTPException
from loguru import logger


from bertrend_apps import SCHEDULER_UTILS
from bertrend_apps.data_provider.data_provider_utils import (
    scrape_feed_from_config,
    scrape,
    auto_scrape,
    generate_query_file,
)
from bertrend_apps.services.config.settings import get_config
from bertrend_apps.services.utils.logging_utils import get_file_logger
from bertrend_apps.services.models.data_provider_models import (
    ScrapeFeedRequest,
    ScrapeRequest,
    ScrapeResponse,
    AutoScrapeRequest,
    GenerateQueryFileRequest,
    GenerateQueryFileResponse,
)

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
        results = await asyncio.to_thread(
            scrape,
            req.keywords,
            req.provider,
            req.after,
            req.before,
            req.max_results,
            req.save_path,
            req.language,
        )
        return ScrapeResponse(
            stored_path=req.save_path.resolve(), article_count=len(results)
        )
    except Exception as e:
        logger.error(f"Error scrapping data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/auto-scrape",
    response_model=ScrapeResponse,
    summary="Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher (multiple requests from a configuration file: each line of the file shall be compliant with the following format:        <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)",
)
async def auto_scrape_api(req: AutoScrapeRequest):
    try:
        results = await asyncio.to_thread(
            auto_scrape,
            req.requests_file,
            req.max_results,
            req.provider,
            req.save_path,
            req.language,
            req.evaluate_articles_quality,
            req.minimum_quality_level,
        )
        return ScrapeResponse(
            stored_path=req.save_path.resolve(), article_count=len(results)
        )
    except Exception as e:
        logger.error(f"Error scrapping data: {e}")
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
    # Create a unique log file for this call
    user = "" if not req.user else req.user
    model_id = "" if not req.model_id else req.model_id

    logger_id = get_file_logger(id="scrape_feed", user_name=user, model_id=model_id)
    try:
        result_path = await asyncio.to_thread(scrape_feed_from_config, req.feed_cfg)
        # Count the articles in the result file
        from bertrend_apps.data_provider.data_provider_utils import count_articles

        article_count = count_articles(result_path)
        return ScrapeResponse(
            stored_path=result_path.resolve(), article_count=article_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping feed: {str(e)}")
    finally:
        # Remove the logger to prevent writing to this file for other calls
        logger.remove(logger_id)


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
