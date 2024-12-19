#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
from pathlib import Path

from bertrend.utils.config_utils import load_toml_config

BERTREND_CONFIG_PATH = Path(__file__).parent / "bertrend.toml"

# Read config
BERTREND_CONFIG = load_toml_config(BERTREND_CONFIG_PATH)
EMBEDDING_CONFIG = BERTREND_CONFIG["embedding_service"]
LLM_CONFIG = BERTREND_CONFIG["llm_service"]

# Linux command to find the index of the GPU device currently less used than the others
BEST_CUDA_DEVICE = (
    "\`nvidia-smi --query-gpu=index,memory.used --format=csv,nounits | tail -n +2 | sort -t',' -k2 -n  "
    "| head -n 1 | cut -d',' -f1\`"
)

BERTREND_BASE_DIR = os.getenv("BERTREND_BASE_DIR", None)
BASE_PATH = (
    Path(BERTREND_BASE_DIR)
    if BERTREND_BASE_DIR
    else Path(__file__).parent.parent.parent
)

# Base dirs
BASE_DATA_PATH = BASE_PATH / "data"
BASE_CACHE_PATH = BASE_PATH / "cache"
BASE_OUTPUT_PATH = BASE_PATH / "output"

FEED_BASE_PATH = BASE_DATA_PATH / "bertrend" / "feeds"
BERTREND_LOG_PATH = BASE_PATH / "logs" / "bertrend"
BERTREND_LOG_PATH.mkdir(parents=True, exist_ok=True)

# Define directories
DATA_PATH = BASE_DATA_PATH / "bertrend"
OUTPUT_PATH = BASE_OUTPUT_PATH / "bertrend"
CACHE_PATH = BASE_CACHE_PATH / "bertrend"

# Weak signals
MODELS_DIR = CACHE_PATH / "models"
ZEROSHOT_TOPICS_DATA_DIR = CACHE_PATH / "zeroshot_topics_data"
SIGNAL_EVOLUTION_DATA_DIR = CACHE_PATH / "signal_evolution_data"

# Create directories if they do not exist
DATA_PATH.mkdir(parents=True, exist_ok=True)
CACHE_PATH.mkdir(parents=True, exist_ok=True)
