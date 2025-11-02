#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ScrapeRequest(BaseModel):
    keywords: str = Field(..., description="Keywords for data search engine.")
    provider: str = Field(
        default="google",
        description="source for data [arxiv, atom, rss, google, bing, newscatcher]",
    )
    after: Optional[str] = Field(
        default=None, description="date after which to consider news [YYYY-MM-DD]"
    )
    before: Optional[str] = Field(
        default=None, description="date before which to consider news [YYYY-MM-DD]"
    )
    max_results: int = Field(default=50, description="maximum results per request")
    save_path: Optional[Path] = Field(
        default=None, description="Path for writing results (jsonl)"
    )
    language: Optional[str] = Field(default=None, description="Language filter")


class ScrapeResponse(BaseModel):
    stored_path: Optional[Path]
    article_count: int


class AutoScrapeRequest(BaseModel):
    requests_file: str = Field(
        ..., description="Path of input file containing the expected queries."
    )
    max_results: int = Field(default=50)
    provider: str = Field(
        default="google",
        description="source for news [arxiv, atom, rss, google, bing, newscatcher]",
    )
    save_path: Optional[Path] = None
    language: Optional[str] = None
    evaluate_articles_quality: bool = False
    minimum_quality_level: str = Field(default="AVERAGE")


class GenerateQueryFileRequest(BaseModel):
    keywords: str
    after: str
    before: str
    save_path: Path
    interval: int = Field(default=30, description="Range of days of atomic requests")


class GenerateQueryFileResponse(BaseModel):
    save_path: Path
    line_count: int


class ScrapeFeedRequest(BaseModel):
    feed_cfg: Path
    user: Optional[str] = None
    model_id: Optional[str] = None
