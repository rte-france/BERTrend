#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""Live integration test for DeepResearchProvider.

Requires a valid OPENAI_API_KEY environment variable. Performs real API calls.

Usage:
    python -m bertrend.tests.data_provider.test_deep_research_live

This script demonstrates the multi-step research pipeline (PLAN → RESEARCH → SYNTHESIZE)
with real-time logging of each step.
"""

import sys

from loguru import logger

# Configure loguru to show INFO+ to stderr in real time
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

from bertrend.bertrend_apps.data_provider.deep_research_provider import DeepResearchProvider

QUERY = "RTE France raccordement parcs éoliens offshore en mer du Nord"
MODEL = "gpt-4.1-mini"
AFTER = "2025-06-01"
BEFORE = "2026-02-10"
LANGUAGE = "fr"
NUM_SUB_QUERIES = 4
SEARCH_CONTEXT_SIZE = "high"


def main():
    provider = DeepResearchProvider(
        model=MODEL,
        search_context_size=SEARCH_CONTEXT_SIZE,
        num_sub_queries=NUM_SUB_QUERIES,
    )

    print("\n" + "=" * 70)
    print("DEEP RESEARCH PROVIDER — LIVE TEST")
    print("=" * 70)
    print(f"Model:       {MODEL}")
    print(f"Sub-queries: {NUM_SUB_QUERIES}")
    print(f"Context:     {SEARCH_CONTEXT_SIZE}")
    print(f"Query:       {QUERY}")
    print(f"Date range:  {AFTER} → {BEFORE}")
    print(f"Language:    {LANGUAGE}")
    print("=" * 70 + "\n")

    articles = provider.get_articles(
        query=QUERY,
        after=AFTER,
        before=BEFORE,
        max_results=1,
        language=LANGUAGE,
    )

    print("\n" + "=" * 70)
    if not articles:
        print("ERROR: No articles returned. Check your OPENAI_API_KEY and model access.")
        sys.exit(1)

    for a in articles:
        print(f"TITLE: {a['title']}")
        print("=" * 70)
        print(f"\nSUMMARY:\n{a['summary']}")
        print(f"\nPRIMARY SOURCE: {a['url']}")
        print(f"TIMESTAMP: {a['timestamp']}")
        print(f"\n{'=' * 70}")
        print("FULL REPORT:\n")
        print(a["text"])
        print("=" * 70)


if __name__ == "__main__":
    main()
