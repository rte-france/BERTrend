#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from datetime import date
from typing import Optional

from pydantic import BaseModel

STRONG_TOPIC_TYPE = "strong"
WEAK_TOPIC_TYPE = "weak"
NOISE_TOPIC_TYPE = "noise"


class Article(BaseModel):
    """Article model."""

    title: str
    date: Optional[date]
    summary: str | None = None
    source: str | None = None
    url: str | None = None


class Topic(BaseModel):
    """Topic model."""

    title: str
    hashtags: list[str] | None = None
    summary: str | None = None
    articles: list[Article]
    topic_type: str = (
        STRONG_TOPIC_TYPE  # Default value "strong", can be "weak" or "noise"
    )


class Newsletter(BaseModel):
    """Newsletter model."""

    title: str
    reference_period: date = None  # has the precedence over the two other dates
    period_start_date: date = None
    period_end_date: date = None
    topics: list[Topic]
    debug_info: dict | None = None
