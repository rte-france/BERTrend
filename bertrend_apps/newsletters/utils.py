#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import glob
import os
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from loguru import logger
from numpy import ndarray

from bertrend import load_toml_config, FEED_BASE_PATH, OUTPUT_PATH
from bertrend.BERTopicModel import BERTopicModel
from bertrend.config.parameters import BERTOPIC_SERIALIZATION
from bertrend.utils.data_loading import TEXT_COLUMN, load_data

# Learning strategies
LEARN_FROM_SCRATCH = (
    "learn_from_scratch"  # uses all available data from feed to create the model
)
LEARN_FROM_LAST = "learn_from_last"  # only the last feed data to create the model
INFERENCE_ONLY = "inference_only"  # do not retrain model; reuse existing bertopic model if available, otherwise, fallback to learn_from_scratch for the first run


def _train_topic_model(
    config_file: Path,
    dataset: pd.DataFrame,
    embedding_model: str,
    embeddings: ndarray,
) -> tuple[list, BERTopic]:
    toml = load_toml_config(config_file)
    # extract relevant bertopic info
    language = toml["bertopic_parameters"].get("language")
    topic_model = BERTopicModel({"global": {"language": language}})
    output = topic_model.fit(
        docs=dataset[TEXT_COLUMN],
        embeddings=embeddings,
        embedding_model=embedding_model,
    )
    return output.topics, output.topic_model


def _load_feed_data(data_feed_cfg: dict, learning_strategy: str) -> pd.DataFrame:
    data_dir = data_feed_cfg["data-feed"].get("feed_dir_path")
    logger.info(f"Loading data from feed dir: {FEED_BASE_PATH / data_dir}")
    # filter files according to extension and pattern
    list_all_files = glob.glob(
        f"{FEED_BASE_PATH}/{data_dir}/*{data_feed_cfg['data-feed'].get('id')}*.jsonl*"
    )
    latest_file = max(list_all_files, key=os.path.getctime)

    if learning_strategy == INFERENCE_ONLY or learning_strategy == LEARN_FROM_LAST:
        # use the last data available in the feed dir
        return load_data(Path(latest_file))

    elif learning_strategy == LEARN_FROM_SCRATCH:
        # use all data available in the feed dir
        dfs = [load_data(Path(f)) for f in list_all_files]
        new_df = pd.concat(dfs).drop_duplicates(
            subset=["title"], keep="first", inplace=False
        )
        return new_df


def _load_topic_model(model_path_dir: str):
    loaded_model = BERTopic.load(model_path_dir)
    return loaded_model


def _save_topic_model(
    topic_model: BERTopic, embedding_model: str, model_path_dir: Path
):
    full_model_path_dir = OUTPUT_PATH / "models" / model_path_dir
    full_model_path_dir.mkdir(parents=True, exist_ok=True)

    # Serialization using safetensors
    topic_model.save(
        full_model_path_dir,
        serialization=BERTOPIC_SERIALIZATION,
        save_ctfidf=True,
        save_embedding_model=embedding_model,
    )
