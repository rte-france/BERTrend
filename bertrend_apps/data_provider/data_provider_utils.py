#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from bertrend import load_toml_config, FEED_BASE_PATH
from bertrend.article_scoring.article_scoring import QualityLevel
from bertrend_apps.common.date_utils import daterange
from bertrend_apps.data_provider.arxiv_provider import ArxivProvider
from bertrend_apps.data_provider.atom_feed_provider import ATOMFeedProvider
from bertrend_apps.data_provider.bing_news_provider import BingNewsProvider
from bertrend_apps.data_provider.google_news_provider import GoogleNewsProvider
from bertrend_apps.data_provider.newscatcher_provider import NewsCatcherProvider
from bertrend_apps.data_provider.rss_feed_provider import RSSFeedProvider

PROVIDERS = {
    "arxiv": ArxivProvider,
    "atom": ATOMFeedProvider,
    "rss": RSSFeedProvider,
    "google": GoogleNewsProvider,
    "bing": BingNewsProvider,
    "newscatcher": NewsCatcherProvider,
}


def scrape(
    keywords: str,
    provider: str,
    after: str,
    before: str,
    max_results: int,
    save_path: Path,
    language: str,
):
    """Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing or NewsCatcher news (single request)."""
    provider_class = PROVIDERS.get(provider)
    provider_instance = provider_class()
    results = provider_instance.get_articles(
        keywords, after, before, max_results, language
    )
    provider_instance.store_articles(results, save_path)
    return results


def auto_scrape(
    requests_file: str,
    max_results: int,
    provider: str,
    save_path: Path,
    language: str,
    evaluate_articles_quality: bool,
    minimum_quality_level,
):
    """Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher (multiple requests)."""
    provider_class = PROVIDERS.get(provider)
    provider_instance = provider_class()
    logger.info(f"Opening query file: {requests_file}")
    with open(requests_file) as file:
        try:
            requests = [line.rstrip().split(";") for line in file]
        except:
            logger.error("Bad file format")
            return -1
        results = provider_instance.get_articles_batch(
            queries_batch=requests,
            max_results=max_results,
            language=language,
            evaluate_articles_quality=evaluate_articles_quality,
            minimum_quality_level=(
                minimum_quality_level
                if isinstance(minimum_quality_level, QualityLevel)
                else QualityLevel.from_string(minimum_quality_level)
            ),
        )
        logger.info(f"Storing {len(results)} articles")
        provider_instance.store_articles(results, save_path)
        return results


def generate_query_file(
    keywords: str,
    after: str,
    before: str,
    interval: int,
    save_path: Path,
):
    """Generates a query file to be used with the auto-scrape command."""
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(after, date_format)
    end_date = datetime.strptime(before, date_format)
    dates_l = list(daterange(start_date, end_date, interval))

    line_count = 0
    with open(save_path, "a") as query_file:
        for elem in dates_l:
            query_file.write(
                f"{keywords};{elem[0].strftime(date_format)};{elem[1].strftime(date_format)}\n"
            )
            line_count += 1
    return line_count


def count_articles(file_path: Path) -> int:
    """Count the number of articles in a JSONL file."""
    try:
        with open(file_path, "r") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def scrape_feed_from_config(
    feed_cfg: Path,
) -> Path:
    """Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher on the basis of a feed configuration file"""
    data_feed_cfg = load_toml_config(feed_cfg)
    current_date = datetime.today()
    current_date_str = current_date.strftime("%Y-%m-%d")
    days_to_subtract = data_feed_cfg["data-feed"].get("number_of_days")
    provider = data_feed_cfg["data-feed"].get("provider")
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
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate a query file
    with tempfile.NamedTemporaryFile() as query_file:
        if (
            provider == "arxiv" or provider == "atom" or provider == "rss"
        ):  # already returns batches
            scrape(
                keywords=keywords,
                provider=provider,
                after=after,
                before=before,
                max_results=max_results,
                save_path=save_path,
                language=language,
            )
        else:
            generate_query_file(
                keywords, after, before, interval=1, save_path=Path(query_file.name)
            )
            auto_scrape(
                requests_file=query_file.name,
                max_results=max_results,
                provider=provider,
                save_path=save_path,
                language=language,
                evaluate_articles_quality=evaluate_articles_quality,
                minimum_quality_level=minimum_quality_level,
            )

    return save_path
