#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import bertrend_apps.data_provider.data_provider_utils as dp_main
from bertrend_apps.data_provider.data_provider_utils import PROVIDERS
from bertrend_apps.services.routers import data_provider as svc


class DummyProvider:
    def __init__(self):
        self.stored = None
        self.last_kwargs = None

    # For /scrape
    def get_articles(self, keywords, after, before, max_results, language):
        # return 3 dummy articles
        self.last_kwargs = {
            "keywords": keywords,
            "after": after,
            "before": before,
            "max_results": max_results,
            "language": language,
        }
        return [
            {"title": "a"},
            {"title": "b"},
            {"title": "c"},
        ]

    # For /auto-scrape and batch flows
    def get_articles_batch(
        self,
        *,
        queries_batch,
        max_results,
        language,
        evaluate_articles_quality,
        minimum_quality_level,
    ):
        # record inputs and return 2 dummy articles
        self.last_kwargs = {
            "queries_batch": queries_batch,
            "max_results": max_results,
            "language": language,
            "evaluate_articles_quality": evaluate_articles_quality,
            "minimum_quality_level": minimum_quality_level,
        }
        return [{"title": "x"}, {"title": "y"}]

    def store_articles(self, results, save_path):
        # just record what would be stored
        self.stored = (len(results), Path(save_path) if save_path else None)
        # Actually write the file for count_articles to work
        if save_path:
            import json

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")


@pytest.fixture
def client(monkeypatch):
    # Mock SEMANTIC_SCHOLAR_API_KEY to handle cases where it's not set
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "mock_api_key_for_testing")

    # Ensure a fresh providers mapping with our dummy for relevant providers
    providers = dict(PROVIDERS)
    providers.update(
        {
            "google": DummyProvider,
            "arxiv": DummyProvider,
            "atom": DummyProvider,
            "rss": DummyProvider,
            "bing": DummyProvider,
            "newscatcher": DummyProvider,
        }
    )
    monkeypatch.setattr(dp_main, "PROVIDERS", providers)
    # Wrap the router in a FastAPI app for proper exception handling
    app = FastAPI()
    app.include_router(svc.router)
    return TestClient(app)


def test_scrape_success(client, tmp_path, monkeypatch):
    # Arrange
    save_path = tmp_path / "out.jsonl"

    # Act
    resp = client.post(
        "/scrape",
        json={
            "keywords": "ai",
            "provider": "google",
            "after": "2025-01-01",
            "before": "2025-01-31",
            "max_results": 10,
            "save_path": str(save_path),
            "language": "en",
        },
    )

    # Assert
    assert resp.status_code == 200
    data = resp.json()
    assert data["article_count"] == 3
    assert Path(data["stored_path"]) == save_path


def test_scrape_unknown_provider(client):
    resp = client.post(
        "/scrape",
        json={
            "keywords": "ai",
            "provider": "unknown",
        },
    )
    assert resp.status_code == 500


def test_auto_scrape_success(client, tmp_path):
    # Prepare a valid request file with two lines
    qf = tmp_path / "queries.txt"
    qf.write_text("kw;2025-01-01;2025-01-02\nkw2;2025-01-03;2025-01-04\n")

    resp = client.post(
        "/auto-scrape",
        json={
            "requests_file": str(qf),
            "max_results": 5,
            "provider": "google",
            "save_path": str(tmp_path / "out.jsonl"),
            "language": "en",
            "evaluate_articles_quality": True,
            "minimum_quality_level": "GOOD",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    # our dummy returns 2 results
    assert data["article_count"] == 2


def test_auto_scrape_bad_file_format(client, tmp_path):
    # Use a directory path to trigger open() failure
    bad_path = tmp_path / "dir_as_file"
    bad_path.mkdir()

    resp = client.post(
        "/auto-scrape",
        json={
            "requests_file": str(bad_path),
            "provider": "google",
        },
    )

    assert resp.status_code == 500


def test_generate_query_file(tmp_path, client):
    # 10-day period with interval 5 should yield 2 lines
    save_path = tmp_path / "queries.txt"
    resp = client.post(
        "/generate-query-file",
        json={
            "keywords": "climate",
            "after": "2025-01-01",
            "before": "2025-01-11",
            "save_path": str(save_path),
            "interval": 5,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert Path(payload["save_path"]) == save_path
    assert payload["line_count"] == 2
    # verify file contents
    lines = save_path.read_text().splitlines()
    assert len(lines) == 2
    assert all(l.startswith("climate;") for l in lines)


def test_scrape_feed_with_arxiv_provider(tmp_path, client, monkeypatch):
    # Prepare config for an arxiv provider (batch provider path directly via /scrape)
    today = datetime.today()
    current_date_str = today.strftime("%Y-%m-%d")
    cfg = {
        "data-feed": {
            "number_of_days": 3,
            "provider": "arxiv",
            "query": "quantum",
            "max_results": 5,
            "language": "en",
            "feed_dir_path": "feed_dir",
            "id": "FEEDX",
            "evaluate_articles_quality": True,
            "minimum_quality_level": "AVERAGE",
        }
    }
    # Mock config loader and FEED_BASE_PATH
    monkeypatch.setattr(dp_main, "load_toml_config", lambda p: cfg)
    monkeypatch.setattr(dp_main, "FEED_BASE_PATH", tmp_path)

    resp = client.post(
        "/scrape-feed",
        json={
            "feed_cfg": str(tmp_path / "fake_cfg.toml"),
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    # DummyProvider.get_articles returns 3 results via /scrape flow
    assert data["article_count"] == 3
    expected_save = tmp_path / "feed_dir" / f"{current_date_str}_FEEDX.jsonl"
    assert Path(data["stored_path"]) == expected_save


def test_scrape_feed_with_google_provider(tmp_path, client, monkeypatch):
    # Prepare config for a non-batch provider: uses generate_query_file + auto_scrape
    cfg = {
        "data-feed": {
            "number_of_days": 1,
            "provider": "google",
            "query": "ai",
            "max_results": 5,
            "language": "en",
            "feed_dir_path": "feed_dir2",
            "id": "FEEDY",
            "evaluate_articles_quality": False,
            "minimum_quality_level": "GOOD",
        }
    }
    monkeypatch.setattr(dp_main, "load_toml_config", lambda p: cfg)
    monkeypatch.setattr(dp_main, "FEED_BASE_PATH", tmp_path)

    resp = client.post(
        "/scrape-feed",
        json={
            "feed_cfg": str(tmp_path / "fake_cfg.toml"),
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    # auto_scrape path uses DummyProvider.get_articles_batch -> 2 results
    assert data["article_count"] == 2
    # stored_path should be inside the base path
    assert str(data["stored_path"]).startswith(str(tmp_path))


def test_schedule_scrapping_success(client, tmp_path, monkeypatch):
    called = {"ok": False}

    def fake_schedule(path):
        called["ok"] = True
        assert str(path).endswith("cfg.toml")

    monkeypatch.setattr(svc.SCHEDULER_UTILS, "schedule_scrapping", fake_schedule)

    resp = client.post(
        "/schedule-scrapping",
        json={"feed_cfg": str(tmp_path / "cfg.toml")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "scheduled"}
    assert called["ok"] is True


def test_schedule_scrapping_error(client, tmp_path, monkeypatch):
    def boom(_):
        raise RuntimeError("fail")

    monkeypatch.setattr(svc.SCHEDULER_UTILS, "schedule_scrapping", boom)

    resp = client.post(
        "/schedule-scrapping",
        json={"feed_cfg": str(tmp_path / "cfg.toml")},
    )
    assert resp.status_code == 500
    assert resp.json()["detail"] == "fail"
