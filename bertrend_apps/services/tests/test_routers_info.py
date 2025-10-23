#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bertrend_apps.services.routers import info


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked config"""

    # Mock the CONFIG to return predictable values
    class MockConfig:
        number_workers = 4

    monkeypatch.setattr(info, "CONFIG", MockConfig())

    app = FastAPI()
    app.include_router(info.router)
    return TestClient(app)


def test_health_endpoint(client):
    """Test the /health endpoint returns ok status"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_num_workers_endpoint(client):
    """Test the /num_workers endpoint returns configured number of workers"""
    response = client.get("/num_workers")
    assert response.status_code == 200
    assert response.json() == 4
