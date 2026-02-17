#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bertrend.bertrend_apps.services.bertrend.routers import newsletters


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(newsletters.router)
    return TestClient(app)


def test_generate_newsletters_success(client, tmp_path, monkeypatch):
    mock_publish = AsyncMock(return_value="cid-123")
    mock_close = AsyncMock()
    monkeypatch.setattr(
        "bertrend.bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
        mock_publish,
    )
    monkeypatch.setattr(
        "bertrend.bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
        mock_close,
    )

    response = client.post(
        "/generate-newsletters",
        json={
            "newsletter_toml_path": str(tmp_path / "newsletter.toml"),
            "data_feed_toml_path": str(tmp_path / "feed.toml"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["correlation_id"] == "cid-123"
    assert mock_publish.await_count == 1
    request_data = mock_publish.await_args.args[0]
    assert request_data["endpoint"] == "/generate-newsletters"
    assert request_data["method"] == "POST"
    assert str(request_data["json_data"]["newsletter_toml_path"]).endswith(
        "newsletter.toml"
    )
    assert str(request_data["json_data"]["data_feed_toml_path"]).endswith("feed.toml")
    assert mock_close.await_count == 1


def test_generate_newsletters_error(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "bertrend.bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    response = client.post(
        "/generate-newsletters",
        json={
            "newsletter_toml_path": str(tmp_path / "newsletter.toml"),
            "data_feed_toml_path": str(tmp_path / "feed.toml"),
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_schedule_newsletters_success(client, monkeypatch, tmp_path):
    called = {"ok": False}

    def fake_schedule(newsletter_path, feed_path):
        called["ok"] = True
        assert str(newsletter_path).endswith("newsletter.toml")
        assert str(feed_path).endswith("feed.toml")

    monkeypatch.setattr(
        newsletters.SCHEDULER_UTILS, "schedule_newsletter", fake_schedule
    )

    response = client.post(
        "/schedule-newsletters",
        json={
            "newsletter_toml_path": str(tmp_path / "newsletter.toml"),
            "data_feed_toml_path": str(tmp_path / "feed.toml"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "Newsletter scheduling completed successfully"}
    assert called["ok"] is True


def test_schedule_newsletters_error(client, monkeypatch, tmp_path):
    def boom(_, __):
        raise RuntimeError("fail")

    monkeypatch.setattr(newsletters.SCHEDULER_UTILS, "schedule_newsletter", boom)

    response = client.post(
        "/schedule-newsletters",
        json={
            "newsletter_toml_path": str(tmp_path / "newsletter.toml"),
            "data_feed_toml_path": str(tmp_path / "feed.toml"),
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "fail"
