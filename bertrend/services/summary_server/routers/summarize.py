#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import time

from fastapi import APIRouter, HTTPException
from loguru import logger

from bertrend.services.summarizer import Summarizer
from bertrend.services.summary_server.models import SummarizeRequest, SummarizeResponse

router = APIRouter()

SUMMARIZER_REGISTRY: dict[str, dict] = {
    "llm": {
        "class_path": "bertrend.services.summary.chatgpt_summarizer.GPTSummarizer",
        "description": "LLM-based summarizer using OpenAI-compatible API",
        "requires_api_key": True,
        "requires_gpu": False,
    },
    "extractive": {
        "class_path": "bertrend.services.summary.extractive_summarizer.ExtractiveSummarizer",
        "description": "Extractive summarizer using sentence embeddings and LexRank",
        "requires_api_key": False,
        "requires_gpu": True,
    },
    "enhanced": {
        "class_path": "bertrend.services.summary.extractive_summarizer.EnhancedExtractiveSummarizer",
        "description": "Hybrid extractive + LLM summarizer for improved fluency",
        "requires_api_key": True,
        "requires_gpu": True,
    },
}

_summarizer_cache: dict[str, Summarizer] = {}


def _get_summarizer(name: str) -> Summarizer:
    """Lazy-load and cache a summarizer instance by registry name."""
    if name not in _summarizer_cache:
        meta = SUMMARIZER_REGISTRY[name]
        module_path, class_name = meta["class_path"].rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        logger.info(f"Loading summarizer: {name} ({class_name})")
        _summarizer_cache[name] = cls()
        logger.info(f"Summarizer '{name}' loaded successfully")
    return _summarizer_cache[name]


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if request.summarizer_type not in SUMMARIZER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown summarizer_type '{request.summarizer_type}'. "
            f"Available: {list(SUMMARIZER_REGISTRY.keys())}",
        )

    start = time.perf_counter()

    texts = request.text if isinstance(request.text, list) else [request.text]

    summarizer = _get_summarizer(request.summarizer_type)

    kwargs = {
        "max_sentences": request.max_sentences,
        "max_length_ratio": request.max_length_ratio,
        "prompt_language": request.language,
    }
    if request.max_words is not None:
        kwargs["max_words"] = request.max_words

    summaries = summarizer.summarize_batch(article_texts=texts, **kwargs)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return SummarizeResponse(
        summaries=summaries,
        summarizer_type=request.summarizer_type,
        language=request.language,
        processing_time_ms=round(elapsed_ms, 2),
    )
