#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import ast
from pydoc import locate

from datetime import datetime

from google.auth.exceptions import RefreshError
from loguru import logger
from pathlib import Path

from bertrend import OUTPUT_PATH
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.config_utils import load_toml_config
from bertrend.utils.data_loading import (
    TIMESTAMP_COLUMN,
    TEXT_COLUMN,
    split_data,
    TITLE_COLUMN,
)
from bertrend.llm_utils.newsletter_features import (
    generate_newsletter,
    render_newsletter,
)
from bertrend_apps.common.mail_utils import get_credentials, send_email
from bertrend_apps.newsletters.utils import (
    INFERENCE_ONLY,
    LEARN_FROM_SCRATCH,
    _load_feed_data,
    _load_topic_model,
    _train_topic_model,
    _save_topic_model,
)

# Config sections
LEARNING_STRATEGY_SECTION = "learning_strategy"
NEWSLETTER_SECTION = "newsletter"


def process_newsletter(
    newsletter_toml_path: Path,
    data_feed_toml_path: Path,
) -> None:
    """
    Core newsletter generation logic (shared between CLI and FastAPI).
    Returns a dict with newsletter info and output path.
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
    split_data_by_paragraphs = learning_strategy.get("split_data_by_paragraphs", "no")
    if model_path:
        model_path = OUTPUT_PATH / model_path
    if learning_type == INFERENCE_ONLY and (not model_path or not model_path.exists()):
        learning_type = LEARN_FROM_SCRATCH

    logger.debug(f"Learning strategy: {learning_type}")

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

    # Deduplicate using only useful columns
    dataset = dataset.drop_duplicates(subset=[TEXT_COLUMN, TITLE_COLUMN]).reset_index(
        drop=True
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # Embed dataset
    logger.info("Computation of embeddings for new data...")
    embedding_model_name = config["embedding_service"].get("model_name")
    embeddings, _, _ = EmbeddingService(
        model_name=embedding_model_name, local=False
    ).embed(dataset[TEXT_COLUMN])

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
    newsletter = generate_newsletter(
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
        newsletter.debug_info = conf_dict

    # Save newsletter
    output_dir = OUTPUT_PATH / newsletter_params.get("output_directory")
    output_format = newsletter_params.get("output_format")
    output_path = (
        output_dir
        / f"{datetime.today().strftime('%Y-%m-%d')}_{newsletter_params.get('id')}"
        f"_{data_feed_cfg['data-feed'].get('id')}.{output_format}"
    )
    render_newsletter(
        newsletter,
        output_path,
        output_format=output_format,
        language=newsletter_params.get("prompt_language", "fr"),
    )
    logger.info(f"Newsletter exported in {output_format} format: {output_path}")

    # Send newsletter by email
    mail_title = (
        title + f" ({newsletter.period_start_date}/{newsletter.period_end_date})"
    )
    recipients = ast.literal_eval(newsletter_params.get("recipients", "[]"))

    try:
        if recipients:
            credentials = get_credentials()
            with open(output_path, "r") as file:
                content = file.read()
            send_email(
                credentials=credentials,
                subject=mail_title,
                recipients=recipients,
                content=content,
                content_type=output_format,
            )
            logger.info(f"Newsletter sent to: {recipients}")
    except RefreshError as re:
        logger.error(f"Problem with token for email, please regenerate it: {re}")
