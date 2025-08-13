import asyncio
from pathlib import Path

from bertrend.article_scoring.article_scoring import ArticleScore
from bertrend.article_scoring.prompts import ARTICLE_SCORING_PROMPT
from bertrend.llm_utils.agent_utils import (
    BaseAgentFactory,
    AsyncAgentConcurrentProcessor,
    progress_reporter,
)
from bertrend.utils.data_loading import load_data


async def score_articles(articles: list[str]):
    agent = BaseAgentFactory().create_agent(
        name="scoring_agent",
        instructions=ARTICLE_SCORING_PROMPT,
        output_type=ArticleScore,
    )

    # Initialize processor
    processor = AsyncAgentConcurrentProcessor(agent=agent, max_concurrent=10)

    # Uncomment to process multiple items:
    results = await processor.process_list_concurrent(
        articles, progress_callback=progress_reporter, chunk_size=10
    )
    return results


if __name__ == "__main__":

    path = Path("/DSIA/nlp/bertrend/data/feeds/feed_nlp/2025-01-07_feed_nlp.jsonl")
    df = load_data(path)
    print(len(df), df.columns)
    df = df.head(5)[["title", "text"]]
    print(df)
    l = list(df.text)
    results = asyncio.run(score_articles(l))

    assert len(results) == len(df)
    df.eval = results
    print(df.eval)
