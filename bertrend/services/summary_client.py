#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import requests


class SummaryAPIClient:
    """Thin HTTP client for the BERTrend Summarization Service."""

    def __init__(self, url: str = "http://localhost:6465"):
        self.url = url.rstrip("/")

    def health(self) -> dict:
        """Check service health."""
        resp = requests.get(f"{self.url}/health")
        resp.raise_for_status()
        return resp.json()

    def list_summarizers(self) -> list[dict]:
        """List available summarizers and their status."""
        resp = requests.get(f"{self.url}/summarizers")
        resp.raise_for_status()
        return resp.json()

    def summarize(
        self,
        text: str | list[str],
        summarizer_type: str = "llm",
        language: str = "fr",
        max_sentences: int = 3,
        max_words: int | None = None,
        max_length_ratio: float = 0.2,
    ) -> dict:
        """Summarize one or more texts.

        Parameters
        ----------
        text : str | list[str]
            Text(s) to summarize.
        summarizer_type : str
            Summarizer backend: "llm", "extractive", or "enhanced".
        language : str
            Prompt language ("fr" or "en").
        max_sentences : int
            Maximum number of sentences in the summary.
        max_words : int | None
            Maximum number of words (optional).
        max_length_ratio : float
            Maximum ratio of summary length to original.

        Returns
        -------
        dict
            Response with keys: summaries, summarizer_type, language, processing_time_ms.
        """
        payload = {
            "text": text,
            "summarizer_type": summarizer_type,
            "language": language,
            "max_sentences": max_sentences,
            "max_length_ratio": max_length_ratio,
        }
        if max_words is not None:
            payload["max_words"] = max_words

        resp = requests.post(f"{self.url}/summarize", json=payload)
        resp.raise_for_status()
        return resp.json()
