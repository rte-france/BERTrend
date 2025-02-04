#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import itertools
import os
from datetime import datetime
from collections import defaultdict

import arxiv

from loguru import logger
import requests

from bertrend.utils.data_loading import TEXT_COLUMN
from bertrend_apps.data_provider.data_provider import DataProvider
from bertrend_apps.data_provider.utils import wait

PAGE_SIZE = 2000
DELAY_SECONDS = 3
NUM_RETRIES = 10

### Request parameters ###
query = "cat:cs.CL"
max_results = float("inf")

### Parameters Semantic Scholar ###
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
if not SEMANTIC_SCHOLAR_API_KEY:
    raise ValueError("SEMANTIC_SCHOLAR_API_KEY environment variable is not set")

PAPERS_PER_REQUEST = 500  # should not be more than 500
SEMANTIC_SCHOLAR_FIELDS = "title,abstract,citationCount,publicationDate,url"

DATE_FORMAT_YYYYMMDD = "%Y-%m-%d"
DATE_FORMAT_YYYYMMDD_TIME = "%Y-%m-%d %H:%M:%S"


class ArxivProvider(DataProvider):
    """Scientific articles providers for Arxiv"""

    def __init__(self):
        self.client = arxiv.Client(
            page_size=PAGE_SIZE,
            delay_seconds=DELAY_SECONDS,
            num_retries=NUM_RETRIES,
        )

    def get_articles(
        self,
        query: str,
        after: str,
        before: str,
        max_results: int,
        language: str = None,
    ) -> list[dict]:
        """Requests the news data provider, collects a set of URLs to be parsed, return results as json lines.

        Parameters
        ----------
        query: str
            keywords describing the request
        after: str
            date after which to consider articles, formatted as YYYY-MM-DD, currently ignored
        before: str
            date before which to consider articles, formatted as YYYY-MM-DD, currently ignored
        max_results: int
            Maximum number of results per request
        language: str
            Language filter

        Returns
        -------
        A list of dict entries, each one describing an article
        """
        begin = datetime.strptime(after, DATE_FORMAT_YYYYMMDD)
        end = datetime.strptime(before, DATE_FORMAT_YYYYMMDD)

        logger.info(f"Querying Arxiv: {query}")
        entries = list(
            self.client.results(
                arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )
            )
        )
        results = self.process_entries(entries, language)
        # post-filtering by date
        results = [
            res
            for res in results
            if begin
            <= datetime.strptime(res["timestamp"], DATE_FORMAT_YYYYMMDD_TIME)
            <= end
        ]
        # add citations count
        return self.add_citations_count(results)

    def _parse_entry(self, entry: arxiv.Result) -> dict | None:
        """Parses a Arxiv entry"""
        try:
            id = entry.entry_id
            title = entry.title
            published = entry.published.strftime(
                "%Y-%m-%d %H:%M:%S"
            )  # dateparser.parse(entry.published).strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(f"----- Title: {title},\tDate: {published}")
            return {
                "id": id,
                "title": title,
                "summary": entry.summary,
                "text": entry.summary,
                "timestamp": published,
            }
        except Exception as e:
            logger.error(str(e) + f"\nError occurred with parsing of: {entry}")
            return None

    @wait(1)
    def _request_semantic_scholar_chunk(self, chunk: list[dict]):
        """Get information from semantic scholar API per batch of articles IDs"""
        ids_list = ["URL:" + entry["id"] for entry in chunk]
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": SEMANTIC_SCHOLAR_FIELDS},
            json={"ids": ids_list},
            headers={"x-api-key": SEMANTIC_SCHOLAR_API_KEY},
        )
        return [item for item in response.json() if item is not None]

    def add_citations_count(self, entries: list[dict]):
        """Uses the semantic_scholar API to get the number of counts per paper"""
        # split list into chunks and request semantic scholar
        chunks = [
            entries[i : i + PAPERS_PER_REQUEST]
            for i in range(0, len(entries), PAPERS_PER_REQUEST)
        ]
        # request API
        semantic_scholar_items_list = list(
            itertools.chain(
                [self._request_semantic_scholar_chunk(chunk) for chunk in chunks]
            )
        )[0]

        # merge semantic scholar entries with arxiv entries

        # update values of entries
        d = defaultdict(dict)

        for item in semantic_scholar_items_list + entries:
            # careful, order is important, semantic_scholar_items may contain fewer items than entries
            d[item["title"]].update(item)

        # filter out possible missing values
        new_entries = [item for item in list(d.values()) if TEXT_COLUMN in item.keys()]

        return new_entries
