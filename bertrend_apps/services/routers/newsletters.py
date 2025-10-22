#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import ast
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException
from loguru import logger
from google.auth.exceptions import RefreshError

from bertrend import BEST_CUDA_DEVICE, OUTPUT_PATH
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
from bertrend_apps import SCHEDULER_UTILS
from bertrend_apps.common.mail_utils import get_credentials, send_email
from bertrend_apps.newsletters.utils import (
    _load_feed_data,
    _load_topic_model,
    _train_topic_model,
    _save_topic_model,
)
from bertrend_apps.services.config.settings import get_config
from bertrend_apps.services.models.newsletters_models import (
    NewsletterRequest,
    NewsletterResponse,
    ScheduleNewsletterRequest,
)
from pydoc import locate

# Config sections
BERTOPIC_CONFIG_SECTION = "bertopic_parameters"
LEARNING_STRATEGY_SECTION = "learning_strategy"
NEWSLETTER_SECTION = "newsletter"

# Learning strategies
LEARN_FROM_SCRATCH = "learn_from_scratch"
LEARN_FROM_LAST = "learn_from_last"
INFERENCE_ONLY = "inference_only"

# Load the configuration
CONFIG = get_config()

router = APIRouter()


# Endpoints
@router.post(
    "/newsletters",
    response_model=NewsletterResponse,
    summary="Generate newsletter from feed",
)
async def newsletter_from_feed(req: NewsletterRequest):
    """
    Creates a newsletter associated to a data feed.
    """
    try:
        logger.info(
            f"Reading newsletters configuration file: {req.newsletter_toml_path}"
        )

        # Read newsletters & data feed configuration
        config = await asyncio.to_thread(load_toml_config, req.newsletter_toml_path)
        data_feed_cfg = await asyncio.to_thread(
            load_toml_config, req.data_feed_toml_path
        )

        learning_strategy = config[LEARNING_STRATEGY_SECTION]
        newsletter_params = config[NEWSLETTER_SECTION]

        # Read data
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

        original_dataset = await asyncio.to_thread(
            lambda: _load_feed_data(data_feed_cfg, learning_type)
            .reset_index(drop=True)
            .reset_index()
        )

        # Split data by paragraphs if required
        dataset = await asyncio.to_thread(
            lambda: split_data(original_dataset)
            .drop("index", axis=1)
            .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
            .reset_index(drop=True)
            .reset_index()
        )

        # Deduplicate
        dataset = dataset.drop_duplicates(
            subset=[TEXT_COLUMN, TITLE_COLUMN]
        ).reset_index(drop=True)
        logger.info(f"Dataset size: {len(dataset)}")

        # Embed dataset
        logger.info("Computation of embeddings for new data...")
        embedding_model_name = config["embedding_service"].get("model_name")
        embeddings, _, _ = await asyncio.to_thread(
            lambda: EmbeddingService(
                model_name=embedding_model_name, local=False
            ).embed(dataset[TEXT_COLUMN])
        )

        if learning_type == INFERENCE_ONLY:
            # Predict only
            topic_model = await asyncio.to_thread(_load_topic_model, model_path)
            logger.info(f"Topic model loaded from {model_path}")
            topics, _ = await asyncio.to_thread(
                topic_model.transform, dataset[TEXT_COLUMN], embeddings
            )
        else:
            # Train topic model
            topics, topic_model = await asyncio.to_thread(
                _train_topic_model,
                req.newsletter_toml_path,
                dataset,
                embedding_model_name,
                embeddings,
            )
            # Save model
            if model_path:
                logger.info(f"Saving topic model to: {model_path}")
                await asyncio.to_thread(
                    _save_topic_model,
                    topic_model,
                    config["embedding_service"].get("model_name"),
                    model_path,
                )
            logger.debug(f"Number of topics: {len(topic_model.get_topic_info()[1:])}")

        summarizer_class = locate(newsletter_params.get("summarizer_class"))
        openai_model_name = newsletter_params.get("openai_model_name", None)

        # Generate newsletter
        logger.info(f"Generating newsletter...")
        title = newsletter_params.get("title")
        newsletter = await asyncio.to_thread(
            generate_newsletter,
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
        await asyncio.to_thread(
            render_newsletter,
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
                credentials = await asyncio.to_thread(get_credentials)
                with open(output_path, "r") as file:
                    content = file.read()
                await asyncio.to_thread(
                    send_email,
                    credentials=credentials,
                    subject=mail_title,
                    recipients=recipients,
                    content=content,
                    content_type=output_format,
                )
                logger.info(f"Newsletter sent to: {recipients}")
        except RefreshError as re:
            logger.error(f"Problem with token for email, please regenerate it: {re}")

        return NewsletterResponse(
            output_path=output_path, status="Newsletter generated successfully"
        )

    except Exception as e:
        logger.error(f"Error generating newsletter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule-newsletters", summary="Schedule newsletter automation")
async def automate_newsletter(req: ScheduleNewsletterRequest):
    """
    Schedule data scrapping on the basis of a feed configuration file.
    """
    try:
        cuda_devices = req.cuda_devices if req.cuda_devices else BEST_CUDA_DEVICE
        await asyncio.to_thread(
            SCHEDULER_UTILS.schedule_newsletter,
            req.newsletter_toml_cfg_path,
            req.data_feed_toml_cfg_path,
            cuda_devices,
        )
        return {"status": "Newsletter scheduling completed successfully"}
    except Exception as e:
        logger.error(f"Error scheduling newsletter: {e}")
        raise HTTPException(status_code=500, detail=str(e))
