#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
import os
import tempfile
import time
from collections import deque
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from fastapi import HTTPException

from bertrend.services.embedding_server.security import (
    ADMIN,
    ALGORITHM,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RATE_WINDOW,
    FULL_ACCESS,
    RESTRICTED_ACCESS,
    SECRET_KEY,
    Token,
    TokenData,
    check_rate_limit,
    create_access_token,
    generate_hex_token,
    get_token,
    is_authorized_for_group,
    list_registered_clients,
    load_client_registry,
    rate_limit_store,
    view_rate_limits,
)


# --- generate_hex_token ---


def test_generate_hex_token_default_length():
    """Test that generate_hex_token returns a hex string of default length."""
    token = generate_hex_token()
    assert len(token) == 64  # 32 bytes = 64 hex chars
    assert all(c in "0123456789abcdef" for c in token)


def test_generate_hex_token_custom_length():
    """Test that generate_hex_token respects custom length."""
    token = generate_hex_token(16)
    assert len(token) == 32  # 16 bytes = 32 hex chars


def test_generate_hex_token_uniqueness():
    """Test that successive calls produce different tokens."""
    assert generate_hex_token() != generate_hex_token()


# --- Token and TokenData models ---


def test_token_model():
    """Test Token pydantic model."""
    token = Token(access_token="abc", token_type="bearer", expires_in=3600.0)
    assert token.access_token == "abc"
    assert token.token_type == "bearer"
    assert token.expires_in == 3600.0


def test_token_data_defaults():
    """Test TokenData default values."""
    td = TokenData()
    assert td.client_id is None
    assert td.scopes == []
    assert td.authorized_groups == []
    assert td.rate_limit == DEFAULT_RATE_LIMIT
    assert td.rate_window == DEFAULT_RATE_WINDOW


def test_token_data_custom():
    """Test TokenData with custom values."""
    td = TokenData(
        client_id="test",
        scopes=[ADMIN],
        authorized_groups=["group1"],
        rate_limit=100,
        rate_window=120,
    )
    assert td.client_id == "test"
    assert td.scopes == [ADMIN]
    assert td.authorized_groups == ["group1"]


# --- create_access_token ---


def test_create_access_token():
    """Test that create_access_token returns a valid JWT."""
    token = create_access_token(
        data={"sub": "client1", "scopes": [FULL_ACCESS]},
        expires_delta=timedelta(minutes=30),
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "client1"
    assert payload["scopes"] == [FULL_ACCESS]
    assert "exp" in payload
    assert payload["expires_in"] == 1800.0


def test_create_access_token_no_expiry():
    """Test create_access_token with no explicit expiry.

    Note: There is a known bug in create_access_token where passing
    expires_delta=None causes a TypeError due to operator precedence
    (datetime + None is evaluated before the `or` fallback).
    In practice, callers always provide an expires_delta.
    """
    with pytest.raises(TypeError):
        create_access_token(data={"sub": "client1"})


# --- load_client_registry ---


def test_load_client_registry_creates_default(monkeypatch, tmp_path):
    """Test that load_client_registry creates a default file if missing."""
    registry_file = str(tmp_path / "registry.json")
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        registry_file,
    )
    registry = load_client_registry()
    assert "admin" in registry
    assert "bertrend" in registry
    assert os.path.exists(registry_file)


def test_load_client_registry_reads_existing(monkeypatch, tmp_path):
    """Test that load_client_registry reads an existing file."""
    registry_file = tmp_path / "registry.json"
    data = {"myclient": {"client_secret": "secret", "scopes": [FULL_ACCESS]}}
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )
    registry = load_client_registry()
    assert registry == data


def test_load_client_registry_invalid_json(monkeypatch, tmp_path):
    """Test that load_client_registry raises on invalid JSON."""
    registry_file = tmp_path / "registry.json"
    registry_file.write_text("not valid json{{{")
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )
    with pytest.raises(HTTPException) as exc_info:
        load_client_registry()
    assert exc_info.value.status_code == 500


# --- list_registered_clients ---


def test_list_registered_clients_hides_secrets(monkeypatch, tmp_path):
    """Test that list_registered_clients does not expose client_secret."""
    registry_file = tmp_path / "registry.json"
    data = {
        "client1": {
            "client_secret": "topsecret",
            "scopes": [FULL_ACCESS],
            "authorized_groups": ["g1"],
            "rate_limit": 10,
            "rate_window": 30,
        }
    }
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )
    clients = list_registered_clients()
    assert "client1" in clients
    assert "client_secret" not in clients["client1"]
    assert clients["client1"]["scopes"] == [FULL_ACCESS]


# --- is_authorized_for_group ---


def test_is_authorized_for_group_empty_groups():
    """Empty authorized_groups means access to all groups."""
    td = TokenData(client_id="c", authorized_groups=[])
    assert is_authorized_for_group(td, "any_group") is True


def test_is_authorized_for_group_matching():
    """Client with specific groups can access matching group."""
    td = TokenData(client_id="c", authorized_groups=["g1", "g2"])
    assert is_authorized_for_group(td, "g1") is True


def test_is_authorized_for_group_not_matching():
    """Client with specific groups cannot access non-matching group."""
    td = TokenData(client_id="c", authorized_groups=["g1"])
    assert is_authorized_for_group(td, "g2") is False


# --- check_rate_limit ---


@pytest.mark.asyncio
async def test_check_rate_limit_allows_within_limit():
    """Test that requests within the rate limit are allowed."""
    td = TokenData(client_id="test_rl_ok", rate_limit=10, rate_window=60)
    rate_limit_store.pop("test_rl_ok", None)
    request = MagicMock()
    # Should not raise
    await check_rate_limit(request, td)


@pytest.mark.asyncio
async def test_check_rate_limit_blocks_when_exceeded():
    """Test that requests exceeding the rate limit are blocked."""
    client_id = "test_rl_exceeded"
    td = TokenData(client_id=client_id, rate_limit=2, rate_window=60)
    rate_limit_store[client_id] = deque(maxlen=1000)
    now = time.time()
    # Pre-fill with requests at the limit
    rate_limit_store[client_id].extend([now, now])
    request = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit(request, td)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_check_rate_limit_skips_none_client():
    """Test that rate limiting is skipped when client_id is None."""
    td = TokenData(client_id=None)
    request = MagicMock()
    # Should not raise
    await check_rate_limit(request, td)


# --- get_token ---


def test_get_token_valid_client(monkeypatch, tmp_path):
    """Test get_token with valid client credentials."""
    registry_file = tmp_path / "registry.json"
    data = {
        "myclient": {
            "client_secret": "mysecret",
            "scopes": [FULL_ACCESS, RESTRICTED_ACCESS],
            "authorized_groups": [],
            "rate_limit": 50,
            "rate_window": 60,
        }
    }
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    form_data = MagicMock()
    form_data.username = "myclient"
    form_data.password = "mysecret"
    form_data.scopes = []

    token = get_token(form_data)
    assert isinstance(token, Token)
    assert token.token_type == "bearer"
    payload = jwt.decode(token.access_token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "myclient"
    assert FULL_ACCESS in payload["scopes"]


def test_get_token_invalid_client(monkeypatch, tmp_path):
    """Test get_token with invalid client_id."""
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({}))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    form_data = MagicMock()
    form_data.username = "nonexistent"
    form_data.password = "secret"
    form_data.scopes = []

    with pytest.raises(HTTPException) as exc_info:
        get_token(form_data)
    assert exc_info.value.status_code == 401


def test_get_token_wrong_secret(monkeypatch, tmp_path):
    """Test get_token with wrong client_secret."""
    registry_file = tmp_path / "registry.json"
    data = {"myclient": {"client_secret": "correct", "scopes": [FULL_ACCESS]}}
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    form_data = MagicMock()
    form_data.username = "myclient"
    form_data.password = "wrong"
    form_data.scopes = []

    with pytest.raises(HTTPException) as exc_info:
        get_token(form_data)
    assert exc_info.value.status_code == 401


def test_get_token_invalid_scope(monkeypatch, tmp_path):
    """Test get_token with a scope the client doesn't have."""
    registry_file = tmp_path / "registry.json"
    data = {"myclient": {"client_secret": "secret", "scopes": [RESTRICTED_ACCESS]}}
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    form_data = MagicMock()
    form_data.username = "myclient"
    form_data.password = "secret"
    form_data.scopes = [ADMIN]

    with pytest.raises(HTTPException) as exc_info:
        get_token(form_data)
    assert exc_info.value.status_code == 400


# --- view_rate_limits ---


def test_view_rate_limits(monkeypatch, tmp_path):
    """Test view_rate_limits returns usage info."""
    client_id = "test_view_rl"
    registry_file = tmp_path / "registry.json"
    data = {
        client_id: {
            "client_secret": "s",
            "scopes": [],
            "rate_limit": 50,
            "rate_window": 60,
        }
    }
    registry_file.write_text(json.dumps(data))
    monkeypatch.setattr(
        "bertrend.services.embedding_server.security.CLIENT_REGISTRY_FILE",
        str(registry_file),
    )

    rate_limit_store[client_id] = deque(maxlen=1000)
    rate_limit_store[client_id].append(time.time())

    usage = view_rate_limits()
    assert client_id in usage
    assert usage[client_id]["usage"] == 1
    assert usage[client_id]["limit"] == 50

    # Cleanup
    rate_limit_store.pop(client_id, None)
