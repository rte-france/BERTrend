#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import tempfile
from unittest.mock import patch

from bertrend.services.summary_server.config.settings import (
    SummaryAPIConfig,
    get_config,
)


def test_default_config():
    """Test that default config loads correctly."""
    config = get_config()
    assert config.host == "0.0.0.0"
    assert config.port == 6465
    assert config.number_workers == 1


def test_custom_config_file():
    """Test loading config from a custom TOML file."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        f.write(b'host = "127.0.0.1"\nport = 9999\nnumber_workers = 4\n')
        f.flush()
        try:
            with patch.dict(os.environ, {"SUMMARY_API_CONFIG_FILE": f.name}):
                config = get_config()
                assert config.host == "127.0.0.1"
                assert config.port == 9999
                assert config.number_workers == 4
        finally:
            os.unlink(f.name)


def test_config_model_fields():
    """Test SummaryAPIConfig accepts all expected fields."""
    config = SummaryAPIConfig(host="localhost", port=8080, number_workers=2)
    assert config.host == "localhost"
    assert config.port == 8080
    assert config.number_workers == 2
