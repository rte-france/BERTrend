import asyncio
from pathlib import Path

from agents import ModelSettings

from bertrend.article_scoring.article_scoring import ArticleScore
from bertrend.article_scoring.prompts import ARTICLE_SCORING_PROMPT
from bertrend.llm_utils.agent_utils import (
    BaseAgentFactory,
    AsyncAgentConcurrentProcessor,
    progress_reporter,
)
from bertrend.utils.data_loading import load_data

DEFAULT_CHUNK_SIZE = 25
DEFAULT_MAX_CONCURRENT_TASKS = 25


async def score_articles(articles: list[str]):
    agent = BaseAgentFactory().create_agent(
        name="scoring_agent",
        instructions=ARTICLE_SCORING_PROMPT,
        output_type=ArticleScore,
        model_settings=ModelSettings(temperature=0.1),
    )

    # Initialize processor
    processor = AsyncAgentConcurrentProcessor(
        agent=agent, max_concurrent=DEFAULT_MAX_CONCURRENT_TASKS
    )

    # Uncomment to process multiple items:
    results = await processor.process_list_concurrent(
        articles, progress_callback=progress_reporter, chunk_size=DEFAULT_CHUNK_SIZE
    )
    return results


if __name__ == "__main__":

    path = Path("/DSIA/nlp/bertrend/data/feeds/feed_nlp/2025-01-07_feed_nlp.jsonl")
    df = load_data(path)
    print(len(df), df.columns)
    l = list(df.text)
    results = asyncio.run(score_articles(l))

    assert len(results) == len(df)
    df["quality_metrics"] = [r.output if not r.error else None for r in results]
    df["overall_quality"] = df["quality_metrics"].apply(
        lambda x: x.quality_level.name if x else None
    )
    print(df.overall_quality)
    category_percentages = df["overall_quality"].value_counts(normalize=True) * 100
    # Display percentages
    print(category_percentages)
