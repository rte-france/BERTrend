#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from bertrend.article_scoring.article_scoring import ArticleScore, CriteriaScores
from bertrend.llm_utils.agent_utils import ProcessingResult
from bertrend.services.article_scoring_server.scoring_service import app


@pytest.fixture
def client():
    return TestClient(app)


def _make_article_score(
    depth: int = 4, originality: int = 3, relevance_to_rte: int = 5
):
    return ArticleScore(
        scores=CriteriaScores(
            depth_of_reporting=depth / 5,
            originality_and_exclusivity=originality / 5,
            source_quality_and_transparency=0.8,
            accuracy_and_fact_checking_rigor=0.8,
            clarity_and_accessibility=0.8,
            balance_and_fairness=0.8,
            narrative_and_engagement=0.8,
            timeliness_and_relevance=0.8,
            ethical_considerations_and_sensitivity=0.8,
            rte_relevance_and_strategic_impact=relevance_to_rte / 5,
        )
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_articles_success(client):
    mock_results = [
        ProcessingResult(input_index=0, output=_make_article_score()),
        ProcessingResult(
            input_index=1, output=_make_article_score(depth=2), error="timeout"
        ),
    ]

    with patch(
        "bertrend.services.article_scoring_server.routers.scoring.score_articles",
        return_value=mock_results,
    ) as mock_score:
        response = client.post("/score", json={"articles": ["article 1", "article 2"]})

    assert response.status_code == 200
    mock_score.assert_called_once_with(["article 1", "article 2"])

    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["article_index"] == 0
    assert data["results"][0]["quality_metrics"] is not None
    assert data["results"][0]["overall_quality"] == "POOR"
    assert data["results"][0]["error"] is None

    assert data["results"][1]["article_index"] == 1
    assert data["results"][1]["quality_metrics"] is not None
    assert data["results"][1]["overall_quality"] is None
    assert data["results"][1]["error"] == "timeout"


def test_score_articles_backend_exception(client):
    with patch(
        "bertrend.services.article_scoring_server.routers.scoring.score_articles",
        side_effect=RuntimeError("provider down"),
    ):
        response = client.post("/score", json={"articles": ["article 1"]})

    assert response.status_code == 500
    assert "provider down" in response.json()["detail"]
