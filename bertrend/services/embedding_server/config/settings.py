#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import tomllib
from pathlib import Path

from pydantic_settings import BaseSettings

DEFAULT_CONFIG_FILE = Path(__file__).parent / "default_config.toml"


class EmbeddingAPIConfig(BaseSettings):
    host: str
    port: int
    model_name: str
    number_workers: int
    cuda_visible_devices: str


# Use environment variable to specify which config to load
def get_config() -> EmbeddingAPIConfig:
    """
    Return the configuration for the embedding API.
    The configuration is loaded from a toml file.
    The path to the toml file can be specified using the environment variable `EMBEDDING_API_CONFIG_FILE`.
    If the environment variable is not set, the default configuration file is used.
    """
    config_file = os.environ.get("EMBEDDING_API_CONFIG_FILE", None)
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    with open(config_file, "rb") as f:
        return EmbeddingAPIConfig(**tomllib.load(f))
