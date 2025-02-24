#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import pandas as pd
import streamlit

from bertrend import MODELS_DIR, FEED_BASE_PATH, CONFIG_PATH

# Config path for users
CONFIG_FEEDS_BASE_PATH = CONFIG_PATH / "users"
CONFIG_FEEDS_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Models config path
BASE_MODELS_DIR = MODELS_DIR / "users"
INTERPRETATION_PATH = "interpretation"

# some identifiers
NOISE = "noise"
WEAK_SIGNALS = "weak_signals"
STRONG_SIGNALS = "strong_signals"
LLM_TOPIC_DESCRIPTION_COLUMN = "LLM Description"
LLM_TOPIC_TITLE_COLUMN = "LLM Title"
URLS_COLUMN = "URLs"

# Models & analysis
DEFAULT_GRANULARITY = 2
DEFAULT_WINDOW_SIZE = 7

DEFAULT_ANALYSIS_CFG = {
    "model_config": {
        "granularity": DEFAULT_GRANULARITY,
        "window_size": DEFAULT_WINDOW_SIZE,
        "language": "French",
    },
    "analysis_config": {
        "topic_evolution": True,
        "evolution_scenarios": True,
        "multifactorial_analysis": True,
    },
}


def get_user_feed_path(user_name: str, feed_id: str) -> Path:
    feed_path = CONFIG_FEEDS_BASE_PATH / user_name / f"{feed_id}_feed.toml"
    return feed_path


def get_user_models_path(user_name: str, model_id: str) -> Path:
    # Path to previously saved models for those data and this user
    models_path = BASE_MODELS_DIR / user_name / model_id
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


def get_model_cfg_path(user_name: str, model_id: str) -> Path:
    model_cfg_path = CONFIG_FEEDS_BASE_PATH / user_name / f"{model_id}_analysis.toml"
    return model_cfg_path


def get_model_interpretation_path(
    user_name: str, model_id: str, reference_ts: pd.Timestamp
) -> Path:
    return (
        get_user_models_path(user_name=user_name, model_id=model_id)
        / INTERPRETATION_PATH
        / reference_ts.strftime("%Y-%m-%d")
    )
