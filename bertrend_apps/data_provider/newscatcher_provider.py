#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os

import dateparser
from loguru import logger
from newscatcherapi import NewsCatcherApiClient

from bertrend_apps.data_provider.data_provider import DataProvider
from bertrend_apps.data_provider.utils import wait

API_KEY = os.getenv("NEWSCATCHER_API_KEY")
if not API_KEY:
    raise ValueError("NEWSCATCHER_API_KEY environment variable is not set")


class NewsCatcherProvider(DataProvider):
    """News provider for Bing News.
    Limitations:
        - depends on API KEY, with free version, request limited to 1/sec; content history limited to one month
    """

    def __init__(self):
        super().__init__()
        self.api_client = NewsCatcherApiClient(x_api_key=API_KEY)

    @wait(1)
    def get_articles(
        self,
        query: str,
        after: str,
        before: str,
        max_results: int,
        language: str = None,
    ) -> list[dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""

        # Use the API to search articles
        logger.info(f"Querying NewsCatcher: {query}")
        result = self.api_client.get_search(
            q=query, lang="fr", page_size=max_results, from_=after, to_=before
        )

        entries = result["articles"][:max_results]
        return self.process_entries(entries, language)

    def _parse_entry(self, entry: dict) -> dict | None:
        """Parses a NewsCatcher news entry"""
        try:
            link = entry["link"]
            url = link
            summary = entry["summary"]
            published = dateparser.parse(entry["published_date"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            text, title = self._get_text(url=url)
            text = self._filter_out_bad_text(text)
            if text is None or text == "":
                return None
            return {
                "title": title,
                "summary": summary,
                "link": link,
                "url": url,
                "text": text,
                "timestamp": published,
            }
        except Exception as e:
            logger.error(
                str(e) + f"\nError occurred with text parsing of url in : {entry}"
            )
            return None
