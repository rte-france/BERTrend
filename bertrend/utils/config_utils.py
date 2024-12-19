#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
import tomllib
from pathlib import Path


def _resolve_env_variables(config_dict: dict) -> dict:
    """
    Recursively resolve environment variables in a dictionary.
    Enviroment variables must be defined as a string like: $VAR_NAME.
    """
    for key, value in config_dict.items():
        # If the value is a string, check for environment variable
        if isinstance(value, str):
            config_dict[key] = os.path.expandvars(value)
        # If it's a nested dictionary, recurse
        elif isinstance(value, dict):
            config_dict[key] = _resolve_env_variables(value)
    return config_dict


def load_toml_config(config_file_path: Path) -> dict:
    """
    Load a TOML configuration file and resolve environment variables.
    """
    with open(config_file_path, "rb") as f:
        config_dict = tomllib.load(f)
    return _resolve_env_variables(config_dict)
