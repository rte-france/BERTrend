#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from loguru import logger

from bertrend.topic_analysis.data_structure import TopicDescription
from bertrend.topic_analysis.topic_description import get_topic_description


def generate_bertrend_topic_description(
    topic_words: str,
    topic_number: int,
    texts: list[str],
    language_code: str = "fr",
) -> tuple[str, str]:
    """Generates a LLM-based human-readable description of a topic composed of a title and a description (as a dict)"""
    if not texts:
        logger.warning(f"No text found for topic number {topic_number}")
        return None, None

    topic_representation = ", ".join(topic_words.split("_"))  # Get top 10 words

    # Prepare the documents text
    docs_text = "\n\n".join(
        [f"Document {i + 1}: {doc[0:2000]}..." for i, doc in enumerate(texts)]
    )

    topic_description: TopicDescription = get_topic_description(
        topic_representation, docs_text, language_code
    )
    if not topic_description:
        return None, None
    return topic_description.title, topic_description.description
