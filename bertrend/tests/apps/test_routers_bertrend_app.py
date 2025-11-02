#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bertrend_apps.services.routers import bertrend_app


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

    def test_train_new_model_success(self, client, monkeypatch, mock_data):
        """Test successful model training"""

        # Mock the train_new_model function to return success
        def mock_train_new_model(model_id, user_name):
            return {
                "status": "success",
                "message": f"Successfully trained new model for user '{user_name}' and model '{model_id}'",
            }

        # Patch where it's imported in bertrend_app module
        from bertrend_apps.services.routers import bertrend_app as ba_module

        monkeypatch.setattr(ba_module, "train_new_model", mock_train_new_model)

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "test_user" in data["message"]
        assert "test_model" in data["message"]

    def test_train_new_model_no_data(self, client, monkeypatch):
        """Test training when no data is available"""

        # Mock the train_new_model function to return no_data status
        def mock_train_new_model(model_id, user_name):
            return {
                "status": "no_data",
                "message": f"No new data found for model '{model_id}'",
            }

        # Patch where it's imported in bertrend_app module
        from bertrend_apps.services.routers import bertrend_app as ba_module

        monkeypatch.setattr(ba_module, "train_new_model", mock_train_new_model)

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_data"
        assert "No new data found" in data["message"]

    def test_train_new_model_empty_dataframe(self, client, monkeypatch):
        """Test training with empty dataframe"""

        # Mock the train_new_model function to return no_data status
        def mock_train_new_model(model_id, user_name):
            return {
                "status": "no_data",
                "message": f"No new data found for model '{model_id}'",
            }

        # Patch where it's imported in bertrend_app module
        from bertrend_apps.services.routers import bertrend_app as ba_module

        monkeypatch.setattr(ba_module, "train_new_model", mock_train_new_model)

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_data"

    def test_train_new_model_error(self, client, monkeypatch):
        """Test error handling during training"""

        # Mock the train_new_model function to raise an exception
        def mock_train_new_model(model_id, user_name):
            raise RuntimeError("Configuration error")

        # Patch where it's imported in bertrend_app module
        from bertrend_apps.services.routers import bertrend_app as ba_module

        monkeypatch.setattr(ba_module, "train_new_model", mock_train_new_model)

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 500
        assert "Configuration error" in response.json()["detail"]

    def test_train_new_model_with_split_by_paragraph_false(
        self, client, monkeypatch, mock_data
    ):
        """Test training with split_by_paragraph set to False"""

        # Mock the train_new_model function to return success
        def mock_train_new_model(model_id, user_name):
            return {
                "status": "success",
                "message": f"Successfully trained new model for user '{user_name}' and model '{model_id}'",
            }

        # Patch where it's imported in bertrend_app module
        from bertrend_apps.services.routers import bertrend_app as ba_module

        monkeypatch.setattr(ba_module, "train_new_model", mock_train_new_model)

        response = client.post(
            "/train-new-model",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"


class TestRegenerate:
    """Tests for /regenerate endpoint"""

    def test_regenerate_success(self, client, monkeypatch):
        """Test successful model regeneration"""

        def mock_regenerate_models(model_id, user, with_analysis, since):
            pass

        monkeypatch.setattr(bertrend_app, "regenerate_models", mock_regenerate_models)

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "test_user" in data["message"]
        assert "test_model" in data["message"]

    def test_regenerate_with_optional_params(self, client, monkeypatch):
        """Test regeneration with optional parameters"""
        called_with = {}

        def mock_regenerate_models(model_id, user, with_analysis, since):
            called_with["model_id"] = model_id
            called_with["user"] = user
            called_with["with_analysis"] = with_analysis
            called_with["since"] = since

        monkeypatch.setattr(bertrend_app, "regenerate_models", mock_regenerate_models)

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
        assert response.json()["status"] == "success"
        assert called_with["with_analysis"] is False
        assert called_with["since"] == pd.Timestamp("2025-01-01")

    def test_regenerate_without_analysis(self, client, monkeypatch):
        """Test regeneration without LLM analysis"""
        called_with = {}

        def mock_regenerate_models(model_id, user, with_analysis, since):
            called_with["with_analysis"] = with_analysis

        monkeypatch.setattr(bertrend_app, "regenerate_models", mock_regenerate_models)

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
                "with_analysis": False,
            },
        )

        assert response.status_code == 200
        assert called_with["with_analysis"] is False

    def test_regenerate_with_since_date(self, client, monkeypatch):
        """Test regeneration with since date filter"""
        called_with = {}

        def mock_regenerate_models(model_id, user, with_analysis, since):
            called_with["since"] = since

        monkeypatch.setattr(bertrend_app, "regenerate_models", mock_regenerate_models)

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
                "since": "2024-06-01",
            },
        )

        assert response.status_code == 200
        assert called_with["since"] == pd.Timestamp("2024-06-01")

    def test_regenerate_error(self, client, monkeypatch):
        """Test error handling during regeneration"""

        def mock_regenerate_models(model_id, user, with_analysis, since):
            raise RuntimeError("Regeneration failed")

        monkeypatch.setattr(bertrend_app, "regenerate_models", mock_regenerate_models)

        response = client.post(
            "/regenerate",
            json={
                "user": "test_user",
                "model_id": "test_model",
            },
        )

        assert response.status_code == 500
        assert "Regeneration failed" in response.json()["detail"]
