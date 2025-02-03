#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from abc import ABC, abstractmethod

DEFAULT_SUMMARIZATION_RATIO = 0.2
DEFAULT_MAX_SENTENCES = 3
DEFAULT_MAX_WORDS = 50


class Summarizer(ABC):
    """Abstract class common to all summarizer implementations."""

    @abstractmethod
    def generate_summary(
        self,
        article_text: str,
        **kwargs,
    ) -> str:
        """Abstract method to generate summary from article text. To be implemented by subclasses."""
        pass

    def summarize_batch(
        self,
        article_texts: list[str],
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        max_words: int = DEFAULT_MAX_WORDS,
        max_length_ratio: float = DEFAULT_SUMMARIZATION_RATIO,
        prompt_language="fr",
        model_name=None,
    ) -> list[str]:
        """Basic implementation of batch summarization. Can be overridden by subclasses."""
        return [
            self.generate_summary(
                article_text=t,
                max_sentences=max_sentences,
                max_words=max_words,
                max_length_ratio=max_length_ratio,
                prompt_language=prompt_language,
                model_name=model_name,
            )
            for t in article_texts
        ]
