#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bertrend_apps.services.bertrend.routers import bertrend_app


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked dependencies"""
    app = FastAPI()
    app.include_router(bertrend_app.router)
    return TestClient(app)


@pytest.fixture
def mock_data():
    """Create mock data for testing"""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "text": [f"Article {i}" for i in range(10)],
            "title": [f"Title {i}" for i in range(10)],
        }
    )


class TestTrainNewModel:
    """Tests for /train-new-model endpoint"""

    def test_train_new_model_success(self, client, monkeypatch):
        """Test successful model training queuing"""

        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "test_user" in data["message"]
        assert "test_model" in data["message"]
        assert data["message"].endswith("(correlation_id: test_correlation_id)")
        mock_publish.assert_called_once()

    def test_train_new_model_no_data(self, client, monkeypatch):
        """Test training when no data is available (now queued)"""
        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

    def test_train_new_model_empty_dataframe(self, client, monkeypatch):
        """Test training with empty dataframe (now queued)"""
        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

    def test_train_new_model_error(self, client, monkeypatch):
        """Test error handling during training queuing"""

        # Mock QueueManager to raise an exception
        mock_publish = AsyncMock(side_effect=RuntimeError("Queue error"))
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 500
        assert "Queue error" in response.json()["detail"]

    def test_train_new_model_with_split_by_paragraph_false(self, client, monkeypatch):
        """Test training (now queued)"""
        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "queued"


class TestRegenerate:
    """Tests for /regenerate endpoint"""

    def test_regenerate_success(self, client, monkeypatch):
        """Test successful model regeneration queuing"""
        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "test_user" in data["message"]
        assert "test_model" in data["message"]
        mock_publish.assert_called_once()

    def test_regenerate_with_optional_params(self, client, monkeypatch):
        """Test regeneration with optional parameters (now queued)"""
        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
                "with_analysis": False,
                "since": "2025-01-01",
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        # Verify that parameters were passed to publish_request
        call_args = mock_publish.call_args[0][0]
        assert call_args["json_data"]["with_analysis"] is False
        assert call_args["json_data"]["since"] == "2025-01-01"

    def test_regenerate_without_analysis(self, client, monkeypatch):
        """Test regeneration without LLM analysis (now queued)"""
        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
                "with_analysis": False,
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        call_args = mock_publish.call_args[0][0]
        assert call_args["json_data"]["with_analysis"] is False

    def test_regenerate_with_since_date(self, client, monkeypatch):
        """Test regeneration with since date filter (now queued)"""
        # Mock QueueManager
        mock_publish = AsyncMock(return_value="test_correlation_id")
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.close",
            AsyncMock(),
        )

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
                "since": "2024-06-01",
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        call_args = mock_publish.call_args[0][0]
        assert call_args["json_data"]["since"] == "2024-06-01"

    def test_regenerate_error(self, client, monkeypatch):
        """Test error handling during regeneration queuing"""
        # Mock QueueManager to raise an exception
        mock_publish = AsyncMock(side_effect=RuntimeError("Queue error"))
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.publish_request",
            mock_publish,
        )
        monkeypatch.setattr(
            "bertrend_apps.services.queue_management.queue_manager.QueueManager.connect",
            AsyncMock(),
        )

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 500
        assert "Queue error" in response.json()["detail"]
