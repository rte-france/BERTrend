#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from fastapi import APIRouter

from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from typing import List
import asyncio

from fastapi import HTTPException
from loguru import logger

from bertrend import FEED_BASE_PATH, load_toml_config
from bertrend.article_scoring.article_scoring import QualityLevel

from bertrend_apps import SCHEDULER_UTILS
from bertrend_apps.common.date_utils import daterange
from bertrend_apps.services.config.settings import get_config
from bertrend_apps.data_provider.arxiv_provider import ArxivProvider
from bertrend_apps.data_provider.atom_feed_provider import ATOMFeedProvider
from bertrend_apps.data_provider.rss_feed_provider import RSSFeedProvider
from bertrend_apps.data_provider.google_news_provider import GoogleNewsProvider
from bertrend_apps.data_provider.bing_news_provider import BingNewsProvider
from bertrend_apps.data_provider.newscatcher_provider import NewsCatcherProvider
from bertrend_apps.services.models.data_provider_models import (
    ScrapeRequest,
    ScrapeResponse,
    AutoScrapeRequest,
    GenerateQueryFileRequest,
    GenerateQueryFileResponse,
    ScrapeFeedRequest,
    ScheduleScrappingRequest,
)

# Load the configuration
# Providers mapping (same as CLI)
PROVIDERS = {
    "arxiv": ArxivProvider,
    "atom": ATOMFeedProvider,
    "rss": RSSFeedProvider,
    "google": GoogleNewsProvider,
    "bing": BingNewsProvider,
    "newscatcher": NewsCatcherProvider,
}

CONFIG = get_config()
# FastAPI application

router = APIRouter()


# Endpoints
@router.post("/scrape", response_model=ScrapeResponse, summary="Scrape data")
async def scrape(req: ScrapeRequest):
    provider_class = PROVIDERS.get(req.provider)
    if provider_class is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
    provider = provider_class()
    results = await asyncio.to_thread(
        provider.get_articles,
        req.keywords,
        req.after,
        req.before,
        req.max_results,
        req.language,
    )
    await asyncio.to_thread(provider.store_articles, results, req.save_path)
    return ScrapeResponse(stored_path=req.save_path, article_count=len(results))


@router.post(
    "/auto-scrape", response_model=ScrapeResponse, summary="Scrape data to file"
)
async def auto_scrape(req: AutoScrapeRequest):
    provider_class = PROVIDERS.get(req.provider)
    if provider_class is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
    provider = provider_class()

    def read_requests_file():
        with open(req.requests_file) as file:
            return [line.rstrip().split(";") for line in file]

    try:
        requests: List[List[str]] = await asyncio.to_thread(read_requests_file)
    except Exception:
        raise HTTPException(status_code=400, detail="Bad file format")

    results = await asyncio.to_thread(
        provider.get_articles_batch,
        queries_batch=requests,
        max_results=req.max_results,
        language=req.language,
        evaluate_articles_quality=req.evaluate_articles_quality,
        minimum_quality_level=QualityLevel.from_string(req.minimum_quality_level),
    )
    logger.info(f"Storing {len(results)} articles")
    await asyncio.to_thread(provider.store_articles, results, req.save_path)
    return ScrapeResponse(stored_path=req.save_path, article_count=len(results))


@router.post(
    "/generate-query-file",
    response_model=GenerateQueryFileResponse,
    summary="Generate query file",
)
async def generate_query_file(req: GenerateQueryFileRequest):
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(req.after, date_format)
    end_date = datetime.strptime(req.before, date_format)
    dates_l = list(daterange(start_date, end_date, req.interval))

    def write_query_file():
        line_count = 0
        with open(req.save_path, "a") as query_file:
            for elem in dates_l:
                query_file.write(
                    f"{req.keywords};{elem[0].strftime(date_format)};{elem[1].strftime(date_format)}\n"
                )
                line_count += 1
        return line_count

    line_count = await asyncio.to_thread(write_query_file)
    return GenerateQueryFileResponse(save_path=req.save_path, line_count=line_count)


@router.post("/scrape-feed", response_model=ScrapeResponse, summary="Scrape data feed")
async def scrape_from_feed(req: ScrapeFeedRequest):
    data_feed_cfg = await asyncio.to_thread(load_toml_config, req.feed_cfg)
    current_date = datetime.today()
    current_date_str = current_date.strftime("%Y-%m-%d")
    days_to_subtract = data_feed_cfg["data-feed"].get("number_of_days")
    provider_name = data_feed_cfg["data-feed"].get("provider")
    keywords = data_feed_cfg["data-feed"].get("query")
    max_results = data_feed_cfg["data-feed"].get("max_results")
    before = current_date_str
    after = (current_date - timedelta(days=days_to_subtract)).strftime("%Y-%m-%d")
    language = data_feed_cfg["data-feed"].get("language")
    save_path = (
        FEED_BASE_PATH
        / data_feed_cfg["data-feed"].get("feed_dir_path")
        / f"{current_date_str}_{data_feed_cfg['data-feed'].get('id')}.jsonl"
    )
    evaluate_articles_quality = data_feed_cfg["data-feed"].get(
        "evaluate_articles_quality", False
    )
    minimum_quality_level = data_feed_cfg["data-feed"].get(
        "minimum_quality_level", QualityLevel.AVERAGE
    )
    await asyncio.to_thread(save_path.parent.mkdir, parents=True, exist_ok=True)

    article_count = 0
    with tempfile.NamedTemporaryFile() as query_file:
        if provider_name in {"arxiv", "atom", "rss"}:  # already returns batches
            resp = await scrape(
                ScrapeRequest(
                    keywords=keywords,
                    provider=provider_name,
                    after=after,
                    before=before,
                    max_results=max_results,
                    save_path=save_path,
                    language=language,
                )
            )
            article_count = resp.article_count
        else:
            _ = await generate_query_file(
                GenerateQueryFileRequest(
                    keywords=keywords,
                    after=after,
                    before=before,
                    interval=1,
                    save_path=Path(query_file.name),
                )
            )
            resp = await auto_scrape(
                AutoScrapeRequest(
                    requests_file=Path(query_file.name),
                    max_results=max_results,
                    provider=provider_name,
                    save_path=save_path,
                    language=language,
                    evaluate_articles_quality=evaluate_articles_quality,
                    minimum_quality_level=str(minimum_quality_level),
                )
            )
            article_count = resp.article_count

    return ScrapeResponse(stored_path=save_path, article_count=article_count)


@router.post("/schedule-scrapping", summary="Schedule data scrapping")
async def automate_scrapping(req: ScheduleScrappingRequest):
    try:
        await asyncio.to_thread(SCHEDULER_UTILS.schedule_scrapping, req.feed_cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "scheduled"}
