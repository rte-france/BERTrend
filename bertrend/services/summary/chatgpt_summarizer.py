#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from loguru import logger

from bertrend import LLM_CONFIG
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.services.summarizer import DEFAULT_MAX_SENTENCES, Summarizer
from bertrend.services.summary.prompts import (
    SYSTEM_SUMMARY_SENTENCES,
)


class GPTSummarizer(Summarizer):
    """Class that uses the GPT service to provide a summary of a text"""

    def __init__(self, api_key: str = None, base_url: str = None):
        # retrieve chatGPT config
        self.api = OpenAI_Client(
            api_key=LLM_CONFIG["api_key"] if not api_key else api_key,
            base_url=LLM_CONFIG["base_url"] if not base_url else base_url,
            model=LLM_CONFIG["model"],
        )
        logger.debug("GPTSummarizer initialized")

    def generate_summary(
        self,
        article_text: str,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        prompt_language: str = "fr",
        max_article_length: int = 1500,
        model_name: str = None,
        **kwargs,
    ) -> str:
        # Limit input length in case the text is large
        article_text = keep_first_n_words(article_text, max_article_length)

        answer = self.api.generate(
            system_prompt=SYSTEM_SUMMARY_SENTENCES[prompt_language].format(
                num_sentences=max_sentences
            ),
            user_prompt=article_text,
        )
        return answer


def keep_first_n_words(text: str, n: int) -> str:
    """This function keeps the first n words of a text.
    Args:
        text: The text string.
        n: The number of words to keep.
    Returns:
        A string containing the first n words of the text.
    """
    words = text.split()
    if n > len(words):
        return text  # Handle case where n is larger than the number of words
    return " ".join(words[:n])
