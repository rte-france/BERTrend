#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import ast
import glob
import os
from pydoc import locate
from typing import List, Tuple

import pandas as pd
import typer
from datetime import datetime

from bertopic import BERTopic
from google.auth.exceptions import RefreshError
from loguru import logger
from pathlib import Path

from numpy import ndarray

from bertrend import FEED_BASE_PATH, BEST_CUDA_DEVICE, OUTPUT_PATH
from bertrend.parameters import BERTOPIC_SERIALIZATION
from bertrend.services.embedding_service import EmbeddingService
from bertrend.topic_model.topic_model import TopicModel
from bertrend.utils.config_utils import load_toml_config
from bertrend.utils.data_loading import (
    TIMESTAMP_COLUMN,
    TEXT_COLUMN,
    load_data,
    split_data,
)
from bertrend.llm_utils.newsletter_features import (
    generate_newsletter,
    export_md_string,
)
from bertrend_apps.common.mail_utils import get_credentials, send_email
from bertrend_apps.common.crontab_utils import schedule_newsletter

# Config sections
BERTOPIC_CONFIG_SECTION = "bertopic_parameters"
LEARNING_STRATEGY_SECTION = "learning_strategy"
NEWSLETTER_SECTION = "newsletter"

# Learning strategies
LEARN_FROM_SCRATCH = (
    "learn_from_scratch"  # uses all available data from feed to create the model
)
LEARN_FROM_LAST = "learn_from_last"  # only the last feed data to create the model
INFERENCE_ONLY = "inference_only"  # do not retrain model; reuse existing bertopic model if available, otherwise, fallback to learn_from_scratch for the first run

# Ensures to write with +rw for both user and groups
os.umask(0o002)

if __name__ == "__main__":
    app = typer.Typer()

    @app.command("newsletters")
    def newsletter_from_feed(
        newsletter_toml_path: Path = typer.Argument(
            help="Path to newsletters toml config file"
        ),
        data_feed_toml_path: Path = typer.Argument(
            help="Path to data feed toml config file"
        ),
    ):
        """
        Creates a newsletter associated to a data feed.
        """
        logger.info(f"Reading newsletters configuration file: {newsletter_toml_path}")

        # read newsletters & data feed configuration
        config = load_toml_config(newsletter_toml_path)
        data_feed_cfg = load_toml_config(data_feed_toml_path)

        learning_strategy = config[LEARNING_STRATEGY_SECTION]
        newsletter_params = config[NEWSLETTER_SECTION]

        # read data
        logger.info(f"Loading dataset...")
        learning_type = learning_strategy.get("learning_strategy", INFERENCE_ONLY)
        model_path = learning_strategy.get("bertopic_model_path", None)
        split_data_by_paragraphs = learning_strategy.get(
            "split_data_by_paragraphs", "no"
        )
        if model_path:
            model_path = OUTPUT_PATH / model_path
        if learning_type == INFERENCE_ONLY and (
            not model_path or not model_path.exists()
        ):
            learning_type = LEARN_FROM_SCRATCH

        logger.info(f"Learning strategy: {learning_type}")

        original_dataset = (
            _load_feed_data(data_feed_cfg, learning_type)
            .reset_index(drop=True)
            .reset_index()
        )

        # split data by paragraphs if required
        dataset = (
            split_data(original_dataset)
            .drop("index", axis=1)
            .sort_values(
                by=TIMESTAMP_COLUMN,
                ascending=False,
            )
            .reset_index(drop=True)
            .reset_index()
        )
        # Deduplicate using all columns
        dataset = dataset.drop_duplicates()
        logger.info(f"Dataset size: {len(dataset)}")

        # Embed dataset
        logger.info("Computation of embeddings for new data...")
        embedding_model_name = config["embedding_service"].get("model_name")
        embeddings, _, _ = EmbeddingService(model_name=embedding_model_name).embed(
            dataset[TEXT_COLUMN]
        )

        if learning_type == INFERENCE_ONLY:
            # predict only
            topic_model = _load_topic_model(model_path)
            logger.info(f"Topic model loaded from {model_path}")
            topics, _ = topic_model.transform(dataset[TEXT_COLUMN], embeddings)

        else:
            # train topic model with the dataset
            topics, topic_model = _train_topic_model(
                config_file=newsletter_toml_path,
                dataset=dataset,
                embedding_model=embedding_model_name,
                embeddings=embeddings,
            )
            # save model
            if model_path:
                logger.info(f"Saving topic model to: {model_path}")
                _save_topic_model(
                    topic_model,
                    config["embedding_service"].get("model_name"),
                    model_path,
                )

            logger.debug(f"Number of topics: {len(topic_model.get_topic_info()[1:])}")

        summarizer_class = locate(newsletter_params.get("summarizer_class"))

        # If no model_name is given, set default model name to env variable $DEFAULT_MODEL_NAME
        openai_model_name = newsletter_params.get("openai_model_name", None)

        # generate newsletters
        logger.info(f"Generating newsletter...")
        title = newsletter_params.get("title")
        newsletter_md, date_from, date_to = generate_newsletter(
            topic_model=topic_model,
            df=original_dataset,
            topics=topics,
            df_split=dataset if split_data_by_paragraphs else None,
            top_n_topics=newsletter_params.get("top_n_topics"),
            top_n_docs=newsletter_params.get("top_n_docs"),
            newsletter_title=title,
            summarizer_class=summarizer_class,
            summary_mode=newsletter_params.get("summary_mode"),
            prompt_language=newsletter_params.get("prompt_language", "fr"),
            improve_topic_description=newsletter_params.get(
                "improve_topic_description", False
            ),
            openai_model_name=openai_model_name,
        )

        if newsletter_params.get("debug", True):
            conf_dict = {section: dict(config[section]) for section in config.keys()}
            newsletter_md += f"\n\n## Debug: config\n\n{conf_dict} \n\n"

        # Save newsletter
        output_dir = OUTPUT_PATH / newsletter_params.get("output_directory")
        output_format = newsletter_params.get("output_format")
        output_path = (
            output_dir
            / f"{datetime.today().strftime('%Y-%m-%d')}_{newsletter_params.get('id')}"
            f"_{data_feed_cfg['data-feed'].get('id')}.{output_format}"
        )
        export_md_string(newsletter_md, output_path, output_format=output_format)
        logger.info(f"Newsletter exported in {output_format} format: {output_path}")

        # Send newsletter by email
        mail_title = title + f" ({date_from}/{date_to})"
        # string to list conversion for recipients
        recipients = ast.literal_eval(newsletter_params.get("recipients", "[]"))
        try:
            if recipients:
                credentials = get_credentials()
                with open(output_path, "r") as file:
                    # Read the entire contents of the file into a string
                    content = file.read()
                send_email(credentials, mail_title, recipients, content, output_format)
                logger.info(f"Newsletter sent to: {recipients}")
        except RefreshError as re:
            logger.error(f"Problem with token for email, please regenerate it: {re}")

    def _train_topic_model(
        config_file: Path,
        dataset: pd.DataFrame,
        embedding_model: str,
        embeddings: ndarray,
    ) -> Tuple[List, BERTopic]:
        topic_model = TopicModel.from_config(config_file)
        output = topic_model.fit(
            docs=dataset[TEXT_COLUMN],
            embedding_model=embedding_model,
            embeddings=embeddings,
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

    @app.command("schedule-newsletters")
    def automate_newsletter(
        newsletter_toml_cfg_path: Path = typer.Argument(
            help="Path to newsletters toml config file"
        ),
        data_feed_toml_cfg_path: Path = typer.Argument(
            help="Path to data feed toml config file"
        ),
        cuda_devices: str = typer.Option(
            BEST_CUDA_DEVICE, help="CUDA_VISIBLE_DEVICES parameters"
        ),
    ):
        """Schedule data scrapping on the basis of a feed configuration file"""
        schedule_newsletter(
            newsletter_toml_cfg_path, data_feed_toml_cfg_path, cuda_devices
        )

    # Main app
    app()
