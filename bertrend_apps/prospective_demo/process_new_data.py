#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from datetime import timedelta
from pathlib import Path

import pandas as pd
import typer
from jsonlines import jsonlines
from loguru import logger

from bertrend import load_toml_config, FEED_BASE_PATH
from bertrend.BERTopicModel import BERTopicModel
from bertrend.BERTrend import train_new_data, BERTrend
from bertrend.services.embedding_service import EmbeddingService
from bertrend.trend_analysis.weak_signals import analyze_signal
from bertrend.utils.data_loading import (
    load_data,
    split_data,
    group_by_days,
    TEXT_COLUMN,
)
from bertrend_apps.prospective_demo import (
    get_user_feed_path,
    get_user_models_path,
    INTERPRETATION_PATH,
    NOISE,
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    DEFAULT_ANALYSIS_CFG,
    get_model_cfg_path,
    URLS_COLUMN,
)
from bertrend_apps.prospective_demo.llm_utils import generate_bertrend_topic_description

DEFAULT_TOP_K = 5


def load_all_data(model_id: str, user: str, language: str):
    # TODO: to be improved
    cfg_file = get_user_feed_path(user, model_id)
    if not cfg_file.exists():
        logger.error(f"Cannot find/process config file: {cfg_file}")
        return
    cfg = load_toml_config(cfg_file)
    feed_base_dir = cfg["data-feed"]["feed_dir_path"]
    files = list(
        Path(FEED_BASE_PATH, feed_base_dir).glob(
            f"*{cfg['data-feed'].get('id')}*.jsonl*"
        )
    )
    if not files:
        logger.warning(f"No new data for '{model_id}', nothing to do")
        return

    dfs = [load_data(Path(f), language=language) for f in files]
    new_data = pd.concat(dfs).drop_duplicates(
        subset=["title"], keep="first", inplace=False
    )
    return new_data


def get_relevant_model_config(
    model_id: str,
    user: str,
):
    # Load model & analysis config
    model_cfg_path = get_model_cfg_path(user, model_id)
    try:
        model_analysis_cfg = load_toml_config(model_cfg_path)
    except Exception:
        model_analysis_cfg = DEFAULT_ANALYSIS_CFG
    # Extract relevant values
    granularity = model_analysis_cfg["model_config"]["granularity"]
    window_size = model_analysis_cfg["model_config"]["window_size"]
    language = model_analysis_cfg["model_config"]["language"]
    if language not in ["French", "English"]:
        language = "French"
    return granularity, window_size, language


def generate_llm_interpretation(
    bertrend: BERTrend,
    reference_timestamp: pd.Timestamp,
    df: pd.DataFrame,
    df_name: str,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Generate detailed analysis for the top k topics using parallel processing.

    Args:
        bertrend: BERTrend instance
        reference_timestamp: Reference timestamp for analysis
        df: Input DataFrame
        df_name: Name of the DataFrame for output
        output_path: Path to save the results
        top_k: Number of top topics to analyze
    """

    interpretation = []
    for topic in df.sort_values(by=["Latest_Popularity"], ascending=False).head(top_k)[
        "Topic"
    ]:
        summary, analysis = analyze_signal(bertrend, topic, reference_timestamp)
        if not summary or not analysis:
            logger.warning(f"Skipping topic {topic} as analysis of signal failed.")
            continue
        interpretation.append(
            {
                "topic": topic,
                "summary": summary.model_dump_json(),
                "analysis": analysis.model_dump_json(),
            }
        )

    # Save interpretation
    output_file_name = output_path / f"{df_name}_interpretation.jsonl"
    with jsonlines.open(
        output_file_name,
        mode="w",
    ) as writer:
        for item in interpretation:
            writer.write(item)
    logger.success(f"Interpretation saved to: {output_file_name}")


def train_new_model_for_period(
    model_id: str,
    user_name: str,
    new_data: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
):
    logger.info(
        f"Training BERTrend model with new data - user: {user_name}, model_id: {model_id}, reference_timestamp: {reference_timestamp}..."
    )
    # Initialization of embedding service
    # TODO: customize service (lang, etc)
    embedding_service = EmbeddingService(local=False)

    # Path to previously saved models for those data and this user
    bertrend_models_path = get_user_models_path(user_name, model_id)

    # Get relevant model info from config
    granularity, window_size, language = get_relevant_model_config(
        model_id=model_id, user=user_name
    )
    language_code = "fr" if language == "French" else "en"

    # Process new data
    bertrend = train_new_data(
        new_data,
        bertrend_models_path=bertrend_models_path,
        embedding_service=embedding_service,
        language=language,
        granularity=granularity,
    )

    if len(bertrend.doc_groups) < 2:
        # This is generally the case when we have only one model
        return

    # Compute popularities
    bertrend.calculate_signal_popularity()

    # classify last signals
    cut_off_date = new_data["timestamp"].max() - timedelta(days=granularity)
    noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = (
        bertrend.classify_signals(window_size, cut_off_date)
    )

    # LLM-based interpretation
    interpretation_path = (
        bertrend_models_path
        / INTERPRETATION_PATH
        / reference_timestamp.strftime("%Y-%m-%d")
    )
    interpretation_path.mkdir(parents=True, exist_ok=True)
    for df, df_name in zip(
        [noise_topics_df, weak_signal_topics_df, strong_signal_topics_df],
        [NOISE, WEAK_SIGNALS, STRONG_SIGNALS],
    ):
        if not df.empty:
            # enrich signal description with LLM-based topic description
            df[[LLM_TOPIC_TITLE_COLUMN, LLM_TOPIC_DESCRIPTION_COLUMN]] = df.apply(
                lambda row: pd.Series(
                    generate_bertrend_topic_description(
                        topic_words=row["Representation"],
                        topic_number=row["Topic"],
                        texts=row["Documents"],
                        language_code=language_code,
                    )
                ),
                axis=1,
            )

            # Add documents URL
            df = pd.merge(
                df,
                bertrend.merged_df[["Topic", URLS_COLUMN]],
                on="Topic",
                how="left",
            )
            df[URLS_COLUMN] = df[URLS_COLUMN].apply(
                lambda x: list(set(x))
            )  # Removes duplicates within each list

            # FIXME: for some unknown reasons, a few elements in the Documents column are not a str but a
            #  timestamp (the identifier of current model); this generates errors when trying to serialize the
            #  df to parquet. The code snippet below is a workaround to avoid this issue.
            df["Documents"] = df["Documents"].apply(
                lambda l: [x if isinstance(x, str) else "" for x in l]
            )

            output_path = interpretation_path / f"{df_name}.parquet"
            df.to_parquet(output_path)
            logger.success(f"{df_name} saved to: {output_path}")

            # Obtain detailed LLM-based interpretion for signals
            generate_llm_interpretation(
                bertrend,
                reference_timestamp=reference_timestamp,
                df=df,
                df_name=df_name,
                output_path=interpretation_path,
            )


def regenerate_models(model_id: str, user: str, with_analysis: bool = True):
    """Regenerate from scratch (method retrospective) the models associated to the specified model identifier
    for the specified user."""

    # Get relevant model info from config
    granularity, window_size, language = get_relevant_model_config(
        model_id=model_id, user=user
    )

    # Load model config
    df = load_all_data(model_id=model_id, user=user, language=language)
    logger.info(f"Size of dataset: {len(df)}")

    # Split data by paragraphs
    df = split_data(df)

    # Process new data and save models
    # - Group data based on granularity
    grouped_data = group_by_days(df=df, day_granularity=granularity)

    if not with_analysis:
        # Path to saved models
        bertrend_models_path = get_user_models_path(user, model_id)

        # Initialization of embedding service
        # TODO: customize service (lang, etc)
        embedding_service = EmbeddingService(local=False)

        # Train BERTrend
        bertrend = BERTrend(
            topic_model=BERTopicModel({"global": {"language": language}})
        )
        embeddings, _, _ = embedding_service.embed(
            texts=df[TEXT_COLUMN],
        )
        bertrend.train_topic_models(
            grouped_data=grouped_data,
            embedding_model=embedding_service.embedding_model_name,
            embeddings=embeddings,
            bertrend_models_path=bertrend_models_path,
            save_topic_models=True,
        )
        bertrend.save_model(models_path=bertrend_models_path)

        logger.success(
            f"Regenerated models for '{model_id}' from scratch. BERTrend model was built using {len(bertrend.doc_groups)} models/time periods."
        )

    else:  # with analysis
        for ts, df in sorted(grouped_data.items()):
            train_new_model_for_period(
                model_id=model_id,
                user_name=user,
                new_data=df.reset_index(drop=True),
                reference_timestamp=ts,
            )

        logger.success(
            f"Regenerated models for '{model_id}' from scratch with LLM-based analysis. BERTrend model was built using {len(grouped_data)} models/time periods."
        )


if __name__ == "__main__":
    app = typer.Typer()

    @app.command("train-new-model")
    def train_new_model(
        user_name: str = typer.Argument(help="Identifier of the user"),
        model_id: str = typer.Argument(help="ID of the model/data to train"),
    ):
        """Incrementally enrich the BERTrend model with new data"""
        logger.info(f'Processing new data for user "{user_name}" about "{model_id}"...')

        # Get relevant model info from config
        granularity, window_size, language = get_relevant_model_config(
            model_id=model_id, user=user_name
        )

        # Load data for last period
        new_data = load_all_data(model_id=model_id, user=user_name, language=language)
        # filter data according to granularity
        # Calculate the date X days ago
        reference_timestamp = pd.Timestamp(
            new_data["timestamp"].max().date()
        )  # used to identify the last model
        cut_off_date = new_data["timestamp"].max() - timedelta(days=granularity)
        # Filter the DataFrame to keep only the rows within the last X days
        filtered_df = new_data[new_data["timestamp"] >= cut_off_date]

        # Split data by paragraphs
        filtered_df = split_data(filtered_df)

        train_new_model_for_period(
            model_id=model_id,
            user_name=user_name,
            new_data=filtered_df,
            reference_timestamp=reference_timestamp,
        )

    @app.command("regenerate")
    def regenerate(
        user: str = typer.Argument(help="identifier of the user"),
        model_id: str = typer.Argument(
            help="ID of the model to be regenerated from scratch"
        ),
        with_analysis: bool = typer.Option(
            default=True, help="Regenerate LLM analysis (may take time)"
        ),
    ):
        """Regenerate past models from scratch"""
        logger.info(
            f"Regenerating models for user '{user}' about '{model_id}', with analysis: {with_analysis}..."
        )
        regenerate_models(model_id=model_id, user=user, with_analysis=with_analysis)

    # Main app
    app()
