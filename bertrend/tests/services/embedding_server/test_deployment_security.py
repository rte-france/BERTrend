#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bertrend.services.embedding_server import security, start

REPOSITORY_ROOT = Path(__file__).parents[4]
DOCKER_COMPOSE_FILE = REPOSITORY_ROOT / "docker-compose.yml"
PACKAGED_CLIENT_REGISTRY = (
    REPOSITORY_ROOT
    / "bertrend"
    / "services"
    / "embedding_server"
    / "bertrend_client_registry.json"
)


def test_docker_compose_requires_runtime_client_secret():
    """Compose must inject one required secret into server and client containers."""
    compose = DOCKER_COMPOSE_FILE.read_text()
    assignments = re.findall(
        r"^\s*-\s+BERTREND_CLIENT_SECRET=(.+)$",
        compose,
        flags=re.MULTILINE,
    )

    assert len(assignments) == 2
    assert all(value.startswith("${BERTREND_CLIENT_SECRET:?") for value in assignments)


def test_repository_does_not_ship_client_credentials():
    """A source checkout or package must never contain a usable client registry."""
    assert not PACKAGED_CLIENT_REGISTRY.exists()
    assert (
        "bertrend/services/embedding_server/bertrend_client_registry.json"
        in (REPOSITORY_ROOT / ".gitignore").read_text().splitlines()
    )


def test_docker_compose_requires_shared_jwt_signing_key():
    """The embedding workers must receive one required JWT signing key."""
    compose = DOCKER_COMPOSE_FILE.read_text()
    assignments = re.findall(
        r"^\s*-\s+BERTREND_SECRET_KEY=(.+)$",
        compose,
        flags=re.MULTILINE,
    )

    assert len(assignments) == 1
    assert assignments[0].startswith("${BERTREND_SECRET_KEY:?")


def test_docker_compose_exposes_embedding_api_on_loopback_only():
    """The development deployment must not publish embeddings on every interface."""
    compose = DOCKER_COMPOSE_FILE.read_text()

    assert '"127.0.0.1:6464:6464"' in compose


def test_server_validates_security_config_before_starting(monkeypatch):
    """Standalone startup must fail before binding a port when secrets are absent."""
    monkeypatch.delenv("BERTREND_SECRET_KEY", raising=False)
    monkeypatch.delenv("BERTREND_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(security, "CLIENT_REGISTRY_FILE", None)
    run_server = MagicMock()
    monkeypatch.setattr(start.uvicorn, "run", run_server)

    with pytest.raises(RuntimeError, match="BERTREND_SECRET_KEY"):
        start.main()

    run_server.assert_not_called()
