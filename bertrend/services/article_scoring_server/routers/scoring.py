#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import APIRouter, HTTPException

from bertrend.article_scoring.scoring_agent import score_articles
from bertrend.services.article_scoring_server.models import (
    ScoreArticlesRequest,
    ScoreArticlesResponse,
    ScoredArticle,
)

router = APIRouter()


@router.post("/score", response_model=ScoreArticlesResponse)
async def score(request: ScoreArticlesRequest):
    try:
        score_results = await score_articles(request.articles)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to score articles: {e}",
        ) from e

    results = []
    for index, item in enumerate(score_results):
        quality_metrics = item.output.serialize_model() if item.output else None
        overall_quality = (
            item.output.quality_level.name if item.output and not item.error else None
        )
        results.append(
            ScoredArticle(
                article_index=index,
                quality_metrics=quality_metrics,
                overall_quality=overall_quality,
                error=item.error,
            )
        )

    return ScoreArticlesResponse(results=results)
