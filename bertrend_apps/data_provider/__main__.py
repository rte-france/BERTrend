#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os

import typer
from pathlib import Path

from bertrend_apps import SCHEDULER_UTILS
from bertrend_apps.data_provider.data_provider_utils import (
    scrape,
    auto_scrape,
    generate_query_file,
    scrape_feed_from_config,
)

# Ensures to write with +rw for both user and groups
os.umask(0o002)

if __name__ == "__main__":
    app = typer.Typer()

    @app.command("scrape")
    def scrape_cli(
        keywords: str = typer.Argument(help="keywords for data search engine."),
        provider: str = typer.Option(
            "google",
            help="source for data [arxiv, atom, rss, google, bing, newscatcher]",
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
        """Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing or NewsCatcher news (single request).

        Parameters
        ----------
        keywords: str
            query described as keywords
        provider: str
            News data provider. Current authorized values [arxiv, atom, rss, google, bing, newscatcher]
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
        scrape(keywords, provider, after, before, max_results, save_path, language)

    @app.command("auto-scrape")
    def auto_scrape_cli(
        requests_file: str = typer.Argument(
            help="path of jsonlines input file containing the expected queries."
        ),
        max_results: int = typer.Option(
            50, help="maximum number of results per request"
        ),
        provider: str = typer.Option(
            "google",
            help="source for news [arxiv, atom, rss, google, bing, newscatcher]",
        ),
        save_path: Path = typer.Option(None, help="Path for writing results."),
        language: str = typer.Option(None, help="Language filter"),
        evaluate_articles_quality: bool = typer.Option(
            False, help="Evaluate quality of articles (LLM-based)"
        ),
        minimum_quality_level: str = typer.Option(
            default="AVERAGE",
            help="Minimum quality level to consider an article as relevant. (among: poor, fair, average, good, excellent)",
        ),
    ):
        """Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher (multiple requests from a configuration file: each line of the file shall be compliant with the following format:
        <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)

        Parameters
        ----------
        requests_file: str
            Text file containing the list of requests to be processed
        max_results: int
            Maximum number of results per request
        provider: str
            News data provider. Current authorized values [arxiv, atom, rss, google, bing, newscatcher]
        save_path: Path
            Path to the output file (jsonl format)
        language: str
            Language filter

        Returns
        -------

        """
        auto_scrape(
            requests_file,
            max_results,
            provider,
            save_path,
            language,
            evaluate_articles_quality,
            minimum_quality_level,
        )

    @app.command("generate-query-file")
    def generate_query_file_cli(
        keywords: str = typer.Argument(help="keywords for news search engine."),
        after: str = typer.Option(
            None, help="date after which to consider news [format YYYY-MM-DD]"
        ),
        before: str = typer.Option(
            None, help="date before which to consider news [format YYYY-MM-DD]"
        ),
        save_path: Path = typer.Option(
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
        generate_query_file(keywords, after, before, interval, save_path)

    @app.command("scrape-feed")
    def scrape_from_feed(
        feed_cfg: Path = typer.Argument(help="Path of the data feed config file"),
    ):
        """Scrape data from Arxiv, ATOM/RSS feeds, Google, Bing news or NewsCatcher on the basis of a feed configuration file"""
        scrape_feed_from_config(feed_cfg)

    @app.command("schedule-scrapping")
    def automate_scrapping(
        feed_cfg: Path = typer.Argument(help="Path of the data feed config file"),
    ):
        """Schedule data scrapping on the basis of a feed configuration file"""
        SCHEDULER_UTILS.schedule_scrapping(feed_cfg)

    ##################
    app()
