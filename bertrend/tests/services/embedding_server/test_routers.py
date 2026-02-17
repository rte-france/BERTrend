#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bertrend.services.embedding_server.routers import authentication, info


# ===================== Info Router Tests =====================


@pytest.fixture
def info_client(monkeypatch):
    """Create a test client with mocked config for info router."""

    class MockConfig:
        model_name = "test-model"
        number_workers = 4

    monkeypatch.setattr(info, "CONFIG", MockConfig())
    app = FastAPI()
    app.include_router(info.router)
    return TestClient(app)


def test_health_endpoint(info_client):
    """Test the /health endpoint returns ok status."""
    response = info_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_name_endpoint(info_client):
    """Test the /model_name endpoint returns configured model name."""
    response = info_client.get("/model_name")
    assert response.status_code == 200
    assert response.json() == "test-model"


def test_num_workers_endpoint(info_client):
    """Test the /num_workers endpoint returns configured number of workers."""
    response = info_client.get("/num_workers")
    assert response.status_code == 200
    assert response.json() == 4


# ===================== Authentication Router Tests =====================


@pytest.fixture
def auth_client(monkeypatch, tmp_path):
    """Create a test client for authentication router with a test registry."""
    registry_file = tmp_path / "registry.json"
    data = {
        "admin_client": {
            "client_secret": "admin_secret",
            "scopes": ["admin", "full_access", "restricted_access"],
            "authorized_groups": [],
            "rate_limit": 100,
            "rate_window": 60,
        },
        "basic_client": {
            "client_secret": "basic_secret",
            "scopes": ["restricted_access"],
            "authorized_groups": ["group1"],
            "rate_limit": 50,
            "rate_window": 60,
        },
    }
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    app = FastAPI()
    app.include_router(authentication.router)
    return TestClient(app)


def test_token_endpoint_valid(auth_client):
    """Test /token endpoint with valid credentials."""
    response = auth_client.post(
        "/token",
        data={
            "username": "admin_client",
            "password": "admin_secret",
            "scope": "",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0


def test_token_endpoint_invalid_client(auth_client):
    """Test /token endpoint with invalid client."""
    response = auth_client.post(
        "/token",
        data={
            "username": "nonexistent",
            "password": "secret",
            "scope": "",
        },
    )
    assert response.status_code == 401


def test_token_endpoint_wrong_secret(auth_client):
    """Test /token endpoint with wrong secret."""
    response = auth_client.post(
        "/token",
        data={
            "username": "admin_client",
            "password": "wrong_secret",
            "scope": "",
        },
    )
    assert response.status_code == 401


def test_list_clients_requires_admin(auth_client):
    """Test /list_registered_clients requires admin scope."""
    # First get a token with restricted_access only
    response = auth_client.post(
        "/token",
        data={
            "username": "basic_client",
            "password": "basic_secret",
            "scope": "",
        },
    )
    token = response.json()["access_token"]

    # Try to list clients - should fail with 403
    response = auth_client.get(
        "/list_registered_clients",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_list_clients_with_admin(auth_client):
    """Test /list_registered_clients with admin token."""
    # Get admin token
    response = auth_client.post(
        "/token",
        data={
            "username": "admin_client",
            "password": "admin_secret",
            "scope": "",
        },
    )
    token = response.json()["access_token"]

    response = auth_client.get(
        "/list_registered_clients",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    clients = response.json()
    assert "admin_client" in clients
    assert "client_secret" not in clients["admin_client"]


def test_rate_limits_endpoint_with_admin(auth_client):
    """Test /rate-limits endpoint with admin token."""
    response = auth_client.post(
        "/token",
        data={
            "username": "admin_client",
            "password": "admin_secret",
            "scope": "",
        },
    )
    token = response.json()["access_token"]

    response = auth_client.get(
        "/rate-limits",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


def test_rate_limits_endpoint_without_admin(auth_client):
    """Test /rate-limits endpoint without admin scope."""
    response = auth_client.post(
        "/token",
        data={
            "username": "basic_client",
            "password": "basic_secret",
            "scope": "",
        },
    )
    token = response.json()["access_token"]

    response = auth_client.get(
        "/rate-limits",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


# ===================== Embeddings Router Tests =====================


def test_encode_endpoint_requires_auth():
    """Test that /encode endpoint requires authentication."""
    # Import embeddings router with mocked model to avoid loading real model
    with patch(
        "bertrend.services.embedding_server.routers.embeddings.EMBEDDING_MODEL"
    ) as mock_model:
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        from bertrend.services.embedding_server.routers import embeddings

        app = FastAPI()
        app.include_router(embeddings.router)
        client = TestClient(app)

        # Request without auth should fail
        response = client.post(
            "/encode",
            json={"text": "hello", "show_progress_bar": False},
        )
        assert response.status_code == 401


def test_encode_endpoint_with_auth(monkeypatch, tmp_path):
    """Test /encode endpoint with valid full_access token."""
    registry_file = tmp_path / "registry.json"
    data = {
        "embed_client": {
            "client_secret": "embed_secret",
            "scopes": ["full_access", "restricted_access"],
            "authorized_groups": [],
            "rate_limit": 100,
            "rate_window": 60,
        }
    }
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    from bertrend.services.embedding_server.routers import embeddings

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    monkeypatch.setattr(embeddings, "EMBEDDING_MODEL", mock_model)

    app = FastAPI()
    app.include_router(authentication.router)
    app.include_router(embeddings.router)
    client = TestClient(app)

    # Get token
    response = client.post(
        "/token",
        data={
            "username": "embed_client",
            "password": "embed_secret",
            "scope": "",
        },
    )
    token = response.json()["access_token"]

    # Encode
    response = client.post(
        "/encode",
        json={"text": "hello world", "show_progress_bar": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "embeddings" in body
    assert body["embeddings"] == [[0.1, 0.2, 0.3]]
    mock_model.encode.assert_called_once_with("hello world", show_progress_bar=False)


def test_encode_endpoint_restricted_access_denied(monkeypatch, tmp_path):
    """Test /encode endpoint denies restricted_access only clients."""
    registry_file = tmp_path / "registry.json"
    data = {
        "restricted_client": {
            "client_secret": "rsecret",
            "scopes": ["restricted_access"],
            "authorized_groups": [],
            "rate_limit": 100,
            "rate_window": 60,
        }
    }
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    from bertrend.services.embedding_server.routers import embeddings

    mock_model = MagicMock()
    monkeypatch.setattr(embeddings, "EMBEDDING_MODEL", mock_model)

    app = FastAPI()
    app.include_router(authentication.router)
    app.include_router(embeddings.router)
    client = TestClient(app)

    # Get token with restricted_access only
    response = client.post(
        "/token",
        data={
            "username": "restricted_client",
            "password": "rsecret",
            "scope": "",
        },
    )
    token = response.json()["access_token"]

    # Try to encode - should fail with 403
    response = client.post(
        "/encode",
        json={"text": "hello", "show_progress_bar": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
