#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from bertrend.utils.config_utils import (
    _resolve_env_variables,
    load_toml_config,
)


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"VAR_NAME": "test_value", "NUMBER": "42"}):
        yield


def test_resolve_env_variables(mock_env):
    config_dict = {
        "key1": "$VAR_NAME",
        "key2": {"subkey1": "$VAR_NAME", "subkey2": "$NUMBER"},
        "key3": "no_env_var",
    }

    resolved_config = _resolve_env_variables(config_dict)

    assert resolved_config["key1"] == "test_value"
    assert resolved_config["key2"]["subkey1"] == "test_value"
    assert resolved_config["key2"]["subkey2"] == "42"
    assert resolved_config["key3"] == "no_env_var"


def test_load_toml_config(mock_env):
    toml_content = """
    [section]
    key1 = "$VAR_NAME"
    key2 = "$NUMBER"
    """

    # Create a temporary TOML file
    toml_file = Path("test_config.toml")
    toml_file.write_text(toml_content)

    try:
        config = load_toml_config(toml_file)
        assert config["section"]["key1"] == "test_value"
        assert config["section"]["key2"] == "42"
    finally:
        toml_file.unlink()  # Clean up the test file
