#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    text: str | list[str]
    summarizer_type: str = "llm"
    language: str = "fr"
    max_sentences: int = 3
    max_words: int | None = None
    max_length_ratio: float = 0.2


class SummarizeResponse(BaseModel):
    summaries: list[str]
    summarizer_type: str
    language: str
    processing_time_ms: float


class SummarizerInfo(BaseModel):
    name: str
    description: str
    requires_api_key: bool
    requires_gpu: bool
    loaded: bool
