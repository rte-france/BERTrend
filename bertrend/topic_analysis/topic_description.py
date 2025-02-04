#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import json

import pandas as pd
from bertopic import BERTopic
from loguru import logger

from bertrend import LLM_CONFIG
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.topic_analysis.prompts import TOPIC_DESCRIPTION_PROMPT


def generate_topic_description(
    topic_model: BERTopic,
    topic_number: int,
    filtered_docs: pd.DataFrame,
    language_code: str = "fr",
) -> dict:
    """Generates a LLM-based human-readable description of a topic composed of a title and a description (as a dict)"""
    topic_words = topic_model.get_topic(topic_number)
    if not topic_words:
        logger.warning(f"No words found for topic number {topic_number}")
        return {"title": "", "description": ""}

    topic_representation = ", ".join(
        [word for word, _ in topic_words[:10]]
    )  # Get top 10 words

    # Prepare the documents text
    docs_text = "\n\n".join(
        [
            f"Document {i + 1}: {doc.text}..."
            for i, doc in filtered_docs.head(3).iterrows()
        ]
    )

    # Prepare the prompt
    prompt = TOPIC_DESCRIPTION_PROMPT[language_code]

    # logger.debug(f"Prompt for GPT:\n{prompt}")
    try:
        client = OpenAI_Client(
            api_key=LLM_CONFIG["api_key"],
            endpoint=LLM_CONFIG["endpoint"],
            model=LLM_CONFIG["model"],
        )
        answer = client.generate(
            response_format={"type": "json_object"},
            user_prompt=prompt.format(
                topic_representation=topic_representation,
                docs_text=docs_text,
            ),
        )
        return json.loads(answer)
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return f"Error generating description: {str(e)}"
