#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from bertrend.llm_utils.newsletter_model import Newsletter, Topic
from bertrend.trend_analysis.data_structure import SignalAnalysis, TopicSummaryList


class TopicOverTime(Topic):
    """Topic model."""

    topic_evolution: TopicSummaryList | None = None
    topic_analysis: SignalAnalysis | None = None


class DetailedNewsletter(Newsletter):
    """Newsletter model with detailed information about topic evolution over time."""

    topics: list[TopicOverTime]
