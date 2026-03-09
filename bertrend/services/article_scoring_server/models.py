#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pydantic import BaseModel


class ScoreArticlesRequest(BaseModel):
    articles: list[str]


class ScoredArticle(BaseModel):
    article_index: int
    quality_metrics: dict | None = None
    overall_quality: str | None = None
    error: str | None = None


class ScoreArticlesResponse(BaseModel):
    results: list[ScoredArticle]
