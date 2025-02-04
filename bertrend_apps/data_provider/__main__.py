#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import tempfile
from datetime import timedelta, datetime

import typer
from loguru import logger
from pathlib import Path

from bertrend import FEED_BASE_PATH, load_toml_config
from bertrend_apps.common.crontab_utils import schedule_scrapping
from bertrend_apps.data_provider.arxiv_provider import ArxivProvider
from bertrend_apps.data_provider.bing_news_provider import BingNewsProvider
from bertrend_apps.data_provider.curebot_provider import CurebotProvider
from bertrend_apps.data_provider.google_news_provider import GoogleNewsProvider
from bertrend_apps.data_provider.newscatcher_provider import NewsCatcherProvider

# Ensures to write with +rw for both user and groups
os.umask(0o002)

PROVIDERS = {
    "arxiv": ArxivProvider,
    "curebot": CurebotProvider,
    "google": GoogleNewsProvider,
    "bing": BingNewsProvider,
    "newscatcher": NewsCatcherProvider,
}

if __name__ == "__main__":
    app = typer.Typer()

    @app.command("scrape")
    def scrape(
        keywords: str = typer.Argument(help="keywords for data search engine."),
        provider: str = typer.Option(
            "google", help="source for data [arxiv, google, bing, newscatcher]"
        ),
        after: str = typer.Option(
            None, help="date after which to consider news [format YYYY-MM-DD]"
        ),
        before: str = typer.Option(
            None, help="date before which to consider news [format YYYY-MM-DD]"
        ),
        max_results: int = typer.Option(
            50, help="maximum number of results per request"
        ),
        save_path: Path = typer.Option(
            None, help="Path for writing results. File is in jsonl format."
        ),
        language: str = typer.Option(None, help="Language filter"),
    ):
        """Scrape data from Arxiv, Google, Bing or NewsCatcher news (single request).

        Parameters
        ----------
        keywords: str
            query described as keywords
        provider: str
            News data provider. Current authorized values [google, bing, newscatcher]
        after: str
            "from" date, formatted as YYYY-MM-DD
        before: str
            "to" date, formatted as YYYY-MM-DD
        max_results: int
            Maximum number of results per request
        save_path: Path
            Path to the output file (jsonl format)
        language: str
            Language filter

        Returns
        -------

        """
        provider_class = PROVIDERS.get(provider)
        provider = provider_class()
        results = provider.get_articles(keywords, after, before, max_results, language)
        provider.store_articles(results, save_path)

    @app.command("auto-scrape")
    def auto_scrape(
        requests_file: str = typer.Argument(
            help="path of jsonlines input file containing the expected queries."
        ),
        max_results: int = typer.Option(
            50, help="maximum number of results per request"
        ),
        provider: str = typer.Option(
            "google", help="source for news [google, bing, newscatcher]"
        ),
        save_path: Path = typer.Option(None, help="Path for writing results."),
        language: str = typer.Option(None, help="Language filter"),
    ):
        """Scrape data from Arxiv, Google, Bing news or NewsCatcher (multiple requests from a configuration file: each line of the file shall be compliant with the following format:
        <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)

        Parameters
        ----------
        requests_file: str
            Text file containing the list of requests to be processed
        max_results: int
            Maximum number of results per request
        provider: str
            News data provider. Current authorized values [google, bing, newscatcher]
        save_path: Path
            Path to the output file (jsonl format)
        language: str
            Language filter

        Returns
        -------

        """
        provider_class = PROVIDERS.get(provider)
        provider = provider_class()
        logger.info(f"Opening query file: {requests_file}")
        with open(requests_file) as file:
            try:
                requests = [line.rstrip().split(";") for line in file]
            except:
                logger.error("Bad file format")
                return -1
            results = provider.get_articles_batch(requests, max_results, language)
            logger.info(f"Storing {len(results)} articles")
            provider.store_articles(results, save_path)

    @app.command("generate-query-file")
    def generate_query_file(
        keywords: str = typer.Argument(help="keywords for news search engine."),
        after: str = typer.Option(
            None, help="date after which to consider news [format YYYY-MM-DD]"
        ),
        before: str = typer.Option(
            None, help="date before which to consider news [format YYYY-MM-DD]"
        ),
        save_path: str = typer.Option(
            None, help="Path for writing results. File is in jsonl format."
        ),
        interval: int = typer.Option(30, help="Range of days of atomic requests"),
    ):
        """Generates a query file to be used with the auto-scrape command. This is useful for queries generating many results.
        This will split the broad query into many ones, each one covering an 'interval' (range) in days covered by each atomic
        request.
        If you want to cover several keywords, run the command several times with the same output file.

        Parameters
        ----------
        keywords: str
            query described as keywords
        after: str
            "from" date, formatted as YYYY-MM-DD
        before: str
            "to" date, formatted as YYYY-MM-DD
        save_path: str
            Path to the output file (jsonl format)

        Returns
        -------

        """
        date_format = "%Y-%m-%d"
        start_date = datetime.strptime(after, date_format)
        end_date = datetime.strptime(before, date_format)
        dates_l = list(_daterange(start_date, end_date, interval))

        with open(save_path, "a") as query_file:
            for elem in dates_l:
                query_file.write(
                    f"{keywords};{elem[0].strftime(date_format)};{elem[1].strftime(date_format)}\n"
                )

    def _daterange(start_date, end_date, ndays):
        for n in range(int((end_date - start_date).days / ndays)):
            yield (
                start_date + timedelta(ndays * n),
                start_date + timedelta(ndays * (n + 1)),
            )

    @app.command("scrape-feed")
    def scrape_from_feed(
        feed_cfg: Path = typer.Argument(help="Path of the data feed config file"),
    ):
        """Scrape data from Arxiv, Google, Bing news or NewsCatcher on the basis of a feed configuration file"""
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
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate a query file
        with tempfile.NamedTemporaryFile() as query_file:
            if provider == "arxiv" or provider == "curebot":  # already returns batches
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
                    keywords, after, before, interval=1, save_path=query_file.name
                )
                auto_scrape(
                    requests_file=query_file.name,
                    max_results=max_results,
                    provider=provider,
                    save_path=save_path,
                    language=language,
                )

    @app.command("schedule-scrapping")
    def automate_scrapping(
        feed_cfg: Path = typer.Argument(help="Path of the data feed config file"),
    ):
        """Schedule data scrapping on the basis of a feed configuration file"""
        schedule_scrapping(feed_cfg)

    ##################
    app()
