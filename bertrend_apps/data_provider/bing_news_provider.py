#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import urllib.parse

import dateparser
import feedparser
from loguru import logger

from bertrend_apps.data_provider.data_provider import DataProvider
from bertrend_apps.data_provider.utils import wait

PATTERN = "{QUERY}"


class BingNewsProvider(DataProvider):
    """News provider for Bing News.
    Limitations:
        - since of results limited to 12
        - hard to request specific dates
    """

    URL_ENDPOINT = f"https://www.bing.com/news/search?q={PATTERN}&format=rss&setLang=fr&sortBy=Date"

    def __init__(self):
        super().__init__()

    @wait(0.2)
    def get_articles(
        self,
        query: str,
        after: str,
        before: str,
        max_results: int,
        language: str = None,
    ) -> list[dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines"""
        q = self._build_query(query, after, before)
        logger.info(f"Querying Bing: {q}")
        result = feedparser.parse(q)
        entries = result["entries"][:max_results]
        return self.process_entries(entries, language)

    def _build_query(self, keywords: str, after: str = None, before: str = None) -> str:
        # FIXME: don't know how to use after/before parameters with Bing news queries
        query = self.URL_ENDPOINT.replace(PATTERN, f"{urllib.parse.quote(keywords)}")
        return query

    def _clean_url(self, bing_url) -> str:
        """Clean encoded URLs returned by Bing news such as "http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;tid=649475a6257945d6900378c8310bcfde&amp;url=https%3a%2f%2fwww.lemondeinformatique.fr%2factualites%2flire-avec-schema-gpt-translator-datastax-automatise-la-creation-de-pipelines-de-donnees-90737.html&amp;c=15009376565431680830&amp;mkt=fr-fr" """
        try:
            clean_url = bing_url.split("url=")[1].split("&  ")[0]
            return urllib.parse.unquote(clean_url)
        except IndexError:
            # fallback (the URL does not match the expected pattern)
            return bing_url

    def _parse_entry(self, entry: dict) -> dict | None:
        """Parses a Bing news entry, uses wait decorator to force delay between 2 successive calls"""
        try:
            link = entry["link"]
            url = self._clean_url(link)
            summary = entry["summary"]
            published = dateparser.parse(entry["published"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            text, title = self._get_text(url=link)
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
