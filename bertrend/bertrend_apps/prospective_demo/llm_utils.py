#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from loguru import logger

from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.topic_analysis.data_structure import TopicDescription
from bertrend.topic_analysis.topic_description import get_topic_description


def generate_bertrend_topic_description(
    topic_words: str,
    topic_number: int,
    texts: list[str],
    language_code: str = "fr",
    reasoning_effort: str | None = None,
    openai_client: OpenAI_Client | None = None,
) -> tuple[str, str]:
    """Generates a LLM-based human-readable description of a topic composed of a title and a description (as a dict).

    openai_client, when provided, is reused across calls to avoid opening a new
    HTTP connection pool per topic (recommended in the queue worker).
    """
    if not texts:
        logger.warning(f"No text found for topic number {topic_number}")
        return None, None

    topic_representation = ", ".join(topic_words.split("_"))  # Get top 10 words

    # Prepare the documents text
    docs_text = "\n\n".join(
        [f"Document {i + 1}: {doc[0:2000]}..." for i, doc in enumerate(texts)]
    )

    topic_description: TopicDescription = get_topic_description(
        topic_representation,
        docs_text,
        language_code,
        reasoning_effort=reasoning_effort,
        openai_client=openai_client,
    )
    if not topic_description:
        return None, None
    return topic_description.title, topic_description.description
