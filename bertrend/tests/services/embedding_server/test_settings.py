#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import tempfile

import pytest

from bertrend.services.embedding_server.config.settings import (
    EmbeddingAPIConfig,
    get_config,
)


def test_default_config():
    """Test that get_config loads the default configuration."""
    config = get_config()
    assert isinstance(config, EmbeddingAPIConfig)
    assert config.host == "0.0.0.0"
    assert config.port == 6464
    assert isinstance(config.model_name, str)
    assert len(config.model_name) > 0
    assert config.number_workers == 2
    assert config.cuda_visible_devices == "0"


def test_custom_config_file(monkeypatch):
    """Test that get_config loads a custom configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(
            'host = "127.0.0.1"\n'
            "port = 9999\n"
            'model_name = "test-model"\n'
            "number_workers = 8\n"
            'cuda_visible_devices = "1,2"\n'
        )
        f.flush()
        monkeypatch.setenv("EMBEDDING_API_CONFIG_FILE", f.name)
        config = get_config()
        assert config.host == "127.0.0.1"
        assert config.port == 9999
        assert config.model_name == "test-model"
        assert config.number_workers == 8
        assert config.cuda_visible_devices == "1,2"
    os.unlink(f.name)


def test_invalid_config_file(monkeypatch):
    """Test that get_config raises an error for an invalid config file."""
    monkeypatch.setenv("EMBEDDING_API_CONFIG_FILE", "/nonexistent/path.toml")
    with pytest.raises(FileNotFoundError):
        get_config()
