#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
from pathlib import Path

from bertrend.utils.config_utils import load_toml_config

# Default config files path
BERTOPIC_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "config" / "bertopic_default_config.toml"
)
BERTREND_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "config" / "bertrend_default_config.toml"
)
SERVICES_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "config" / "services_default_config.toml"
)

# Read config
BERTOPIC_CONFIG = load_toml_config(BERTOPIC_DEFAULT_CONFIG_PATH)
BERTREND_CONFIG = load_toml_config(BERTREND_DEFAULT_CONFIG_PATH)
SERVICES_CONFIG = load_toml_config(SERVICES_DEFAULT_CONFIG_PATH)

EMBEDDING_CONFIG = SERVICES_CONFIG["embedding_service"]
LLM_CONFIG = SERVICES_CONFIG["llm_service"]

# Linux command to find the index of the GPU device currently less used than the others
BEST_CUDA_DEVICE = r"\`nvidia-smi --query-gpu=index,memory.used --format=csv,nounits | tail -n +2 | sort -t',' -k2 -n  | head -n 1 | cut -d',' -f1\`"

BERTREND_BASE_DIR = os.getenv("BERTREND_BASE_DIR", None)
BASE_PATH = (
    Path(BERTREND_BASE_DIR)
    if BERTREND_BASE_DIR
    else Path(__file__).parent.parent.parent
)

# Base dirs
DATA_PATH = BASE_PATH / "data"
CACHE_PATH = BASE_PATH / "cache"
OUTPUT_PATH = BASE_PATH / "output"
CONFIG_PATH = BASE_PATH / "config"

FEED_BASE_PATH = DATA_PATH / "feeds"
BERTREND_LOG_PATH = BASE_PATH / "logs"

# Weak signals
MODELS_DIR = CACHE_PATH / "models"
ZEROSHOT_TOPICS_DATA_DIR = CACHE_PATH / "zeroshot_topics_data"
SIGNAL_EVOLUTION_DATA_DIR = CACHE_PATH / "signal_evolution_data"

# Create directories if they do not exist
DATA_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CACHE_PATH.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.mkdir(parents=True, exist_ok=True)
BERTREND_LOG_PATH.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
