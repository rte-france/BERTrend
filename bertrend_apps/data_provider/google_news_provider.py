#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import urllib.parse

import dateparser
from loguru import logger
from pygooglenews import GoogleNews

from bertrend_apps.data_provider.data_provider import DataProvider
from bertrend_apps.data_provider.utils import wait, decode_google_news_url

PATTERN = "{QUERY}"
BEFORE = "+before:today"
AFTER = "+after:2000-01-01"
MAX_ARTICLES = 100


class GoogleNewsProvider(DataProvider):
    """News provider for Google News.
    Limitations:
        - since of results limited to 100
    """

    URL_ENDPOINT = f"https://news.google.com/rss/search?num={MAX_ARTICLES}&hl=fr&gl=FR&ceid=FR:fr&q={PATTERN}{BEFORE}{AFTER}"

    def __init__(self):
        super().__init__()
        self.gn = GoogleNews()

    @wait(1)
    def get_articles(
        self,
        query: str,
        after: str = None,
        before: str = None,
        max_results: int = 50,
        language: str = "fr",
    ) -> list[dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        # FIXME: this may be blocked by google
        if language and language != "en":
            self.gn.lang = language.lower()
            self.gn.country = language.upper()

        logger.info(f"Querying Google: {query}")
        result = self.gn.search(query, from_=after, to_=before)
        entries = result["entries"][:max_results]

        return self.process_entries(entries, language)

    def _build_query(self, keywords: str, after: str = None, before: str = None) -> str:
        query = self.URL_ENDPOINT.replace(PATTERN, f"{urllib.parse.quote(keywords)}")
        if after is None or after == "":
            query = query.replace(AFTER, "")
        else:
            query = query.replace(AFTER, f"+after:{after}")
        if before is None or before == "":
            query = query.replace(BEFORE, "")
        else:
            query = query.replace(BEFORE, f"+before:{before}")

        return query

    def _parse_entry(self, entry: dict) -> dict | None:
        """Parses a Google News entry"""
        try:
            # NB. we do not use the title from Gnews as it is sometimes truncated
            link = entry["link"]
            url = decode_google_news_url(link)
            summary = entry["summary"]
            published = dateparser.parse(entry["published"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            text, title = self._get_text(url=url)
            text = self._filter_out_bad_text(text)
            if text is None or text == "":
                return None
            logger.debug(f"----- Title: {title},\tDate: {published}")
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
