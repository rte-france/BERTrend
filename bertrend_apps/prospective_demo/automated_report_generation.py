#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import pandas as pd
import typer
from google.auth.exceptions import RefreshError
from loguru import logger

from bertrend import load_toml_config
from bertrend.llm_utils.newsletter_model import (
    WEAK_TOPIC_TYPE,
    STRONG_TOPIC_TYPE,
    Article,
)
from bertrend.trend_analysis.data_structure import TopicSummaryList, SignalAnalysis

from bertrend_apps.common.mail_utils import get_credentials, send_email
from bertrend_apps.prospective_demo import (
    get_model_interpretation_path,
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    URLS_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    DEFAULT_ANALYSIS_CFG,
    get_model_cfg_path,
    get_user_models_path,
)
from bertrend_apps.prospective_demo.data_model import DetailedNewsletter, TopicOverTime
from bertrend_apps.prospective_demo.report_generation_utils import (
    render_html_report,
    create_temp_report,
    MAXIMUM_NUMBER_OF_ARTICLES,
)
from bertrend_apps.prospective_demo.utils import is_valid_email


def send_email_automated(
    temp_path: Path,
    mail_title: str,
    recipients: list[str],
    model_id: str,
    reference_date: str,
) -> None:
    """
    Send email automatically without Streamlit UI.

    Args:
        temp_path: Path to the HTML report file
        mail_title: Subject of the email
        recipients: List of email addresses
        model_id: Model identifier
        reference_date: Date string for the report
    """
    if not recipients:
        logger.warning("No recipients specified for automated email sending")
        return

    if not all(is_valid_email(email) for email in recipients):
        logger.error("Invalid email address(es) in recipients list")
        return

    try:
        credentials = get_credentials()
        logger.info(f"Sending automated report to {len(recipients)} recipient(s)")
        send_email(
            credentials=credentials,
            subject=mail_title,
            recipients=recipients,
            content=temp_path,
            file_name=f"{reference_date}_{model_id}.html",
        )
        logger.success(f"Email sent successfully to: {', '.join(recipients)}")
    except RefreshError as re:
        logger.error(f"Problem with token for email, please regenerate it: {re}")
    except Exception as e:
        logger.error(f"Error sending email: {e}")


def load_signal_data(
    user: str,
    model_id: str,
    reference_ts: pd.Timestamp,
    max_emerging_topics: int = None,
    max_strong_topics: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load weak and strong signal data for a given model and timestamp.

    Args:
        user: User identifier
        model_id: Model identifier
        reference_ts: Reference timestamp
        max_emerging_topics: Maximum number of emerging (weak) topics to return
        max_strong_topics: Maximum number of strong topics to return

    Returns:
        Tuple of (weak_signals_df, strong_signals_df)
    """
    interpretation_path = get_model_interpretation_path(user, model_id, reference_ts)

    # Load from parquet files which contain all necessary columns
    # (URLs, LLM Title, LLM Description, Topic, etc.) along with interpretation data
    weak_signals_path = interpretation_path / f"{WEAK_SIGNALS}.parquet"
    strong_signals_path = interpretation_path / f"{STRONG_SIGNALS}.parquet"

    # Also check for interpretation JSONL files
    weak_signals_jsonl_path = (
        interpretation_path / f"{WEAK_SIGNALS}_interpretation.jsonl"
    )
    strong_signals_jsonl_path = (
        interpretation_path / f"{STRONG_SIGNALS}_interpretation.jsonl"
    )

    weak_signals = None
    strong_signals = None

    # Try loading weak signals from parquet first, then JSONL as fallback
    if weak_signals_path.exists():
        weak_signals = pd.read_parquet(weak_signals_path)
        # Merge with interpretation data if available
        if weak_signals_jsonl_path.exists():
            interpretation_df = pd.read_json(weak_signals_jsonl_path, lines=True)
            # Rename 'topic' column to 'Topic' for consistency
            if "topic" in interpretation_df.columns:
                interpretation_df = interpretation_df.rename(columns={"topic": "Topic"})
            # Merge interpretation data with parquet data
            weak_signals = pd.merge(
                weak_signals,
                interpretation_df[["Topic", "summary", "analysis"]],
                on="Topic",
                how="left",
            )
        logger.info(f"Loaded {len(weak_signals)} weak signals")
        if max_emerging_topics is not None and len(weak_signals) > max_emerging_topics:
            weak_signals = weak_signals.head(max_emerging_topics)
            logger.info(f"Limited to {max_emerging_topics} weak signals")
    else:
        logger.warning(f"No weak signals found at {weak_signals_path}")

    # Try loading strong signals from parquet first, then JSONL as fallback
    if strong_signals_path.exists():
        strong_signals = pd.read_parquet(strong_signals_path)
        # Merge with interpretation data if available
        if strong_signals_jsonl_path.exists():
            interpretation_df = pd.read_json(strong_signals_jsonl_path, lines=True)
            # Rename 'topic' column to 'Topic' for consistency
            if "topic" in interpretation_df.columns:
                interpretation_df = interpretation_df.rename(columns={"topic": "Topic"})
            # Merge interpretation data with parquet data
            strong_signals = pd.merge(
                strong_signals,
                interpretation_df[["Topic", "summary", "analysis"]],
                on="Topic",
                how="left",
            )
        logger.info(f"Loaded {len(strong_signals)} strong signals")
        if max_strong_topics is not None and len(strong_signals) > max_strong_topics:
            strong_signals = strong_signals.head(max_strong_topics)
            logger.info(f"Limited to {max_strong_topics} strong signals")
    else:
        logger.warning(f"No strong signals found at {strong_signals_path}")

    return weak_signals, strong_signals


def create_detailed_newsletter_automated(
    weak_signals: pd.DataFrame,
    strong_signals: pd.DataFrame,
    model_id: str,
    reference_date: pd.Timestamp,
    options: dict = None,
) -> DetailedNewsletter:
    """
    Create a detailed newsletter without Streamlit session state.
    Similar to create_detailed_newsletter but for automated use.
    """
    detailed_newsletter = DetailedNewsletter(
        title=model_id,
        reference_period=reference_date.date(),
        topics=[],
    )

    for df, topic_type in zip(
        [weak_signals, strong_signals], [WEAK_TOPIC_TYPE, STRONG_TOPIC_TYPE]
    ):
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            articles = [
                Article(title=link, date=None, source=None, url=link)
                for link in list(set(row[URLS_COLUMN]))[:MAXIMUM_NUMBER_OF_ARTICLES]
            ]

            topic: TopicOverTime = TopicOverTime(
                title=row[LLM_TOPIC_TITLE_COLUMN],
                hashtags=None,
                summary=row[LLM_TOPIC_DESCRIPTION_COLUMN],
                articles=articles,
                topic_type=topic_type,
                topic_evolution=TopicSummaryList.model_validate_json(row["summary"]),
                topic_analysis=SignalAnalysis.model_validate_json(row["analysis"]),
            )

            if options is not None and not options.get("topic_evolution", True):
                topic.topic_evolution = None
            if options is not None and not options.get("evolution_scenarios", True):
                topic.topic_analysis.evolution_scenario = None
            if options is not None and not options.get("multifactorial_analysis", True):
                topic.topic_analysis.potential_implications = None
                topic.topic_analysis.topic_interconnexions = None
                topic.topic_analysis.drivers_inhibitors = None

            detailed_newsletter.topics.append(topic)

    return detailed_newsletter


def generate_automated_report(
    user: str,
    model_id: str,
    reference_date: str = None,
) -> None:
    """
    Generate and send a report automatically based on the model configuration.

    Args:
        user: User identifier
        model_id: Model identifier
        reference_date: Optional date string (YYYY-MM-DD). If None, uses the most recent data.
    """
    logger.info(
        f"Starting automated report generation for user '{user}', model '{model_id}'"
    )

    # Load model configuration
    model_cfg_path = get_model_cfg_path(user, model_id)
    if model_cfg_path.exists():
        model_config = load_toml_config(model_cfg_path)
    else:
        logger.warning(f"Model config not found at {model_cfg_path}, using defaults")
        model_config = DEFAULT_ANALYSIS_CFG

    # Get report configuration
    report_config = model_config.get("report_config", {})
    auto_send = report_config.get("auto_send", False)
    recipients = report_config.get("email_recipients", [])
    report_title = report_config.get("report_title", f"Automated Report - {model_id}")

    if not auto_send:
        logger.warning(
            "auto_send is disabled in configuration. Skipping report generation. (user: '{user}', model: '{model_id}')"
        )
        raise ValueError(
            "auto_send is disabled in configuration. (user: '{user}', model: '{model_id}')"
        )

    if not recipients:
        logger.warning(
            "No email recipients configured. Skipping report generation. (user: '{user}', model: '{model_id}')"
        )
        raise ValueError(
            "No email recipients configured. (user: '{user}', model: '{model_id}')"
        )

    # Determine reference timestamp
    if reference_date:
        reference_ts = pd.Timestamp(reference_date)
    else:
        # Find the most recent interpretation data
        models_path = get_user_models_path(user, model_id)
        interpretation_path = models_path / "interpretation"
        if not interpretation_path.exists():
            logger.error(f"No interpretation data found at {interpretation_path}")
            raise FileNotFoundError(
                f"No interpretation data found at {interpretation_path}"
            )

        # Get the most recent date directory
        date_dirs = sorted([d for d in interpretation_path.iterdir() if d.is_dir()])
        if not date_dirs:
            logger.error(
                "No date directories found in interpretation path. (user: '{user}', model: '{model_id}')"
            )
            raise FileNotFoundError(
                "No date directories found in interpretation path. (user: '{user}', model: '{model_id}')"
            )
        reference_ts = pd.Timestamp(date_dirs[-1].name)

    logger.info(f"Using reference date: {reference_ts.date()}")

    # Get topic limits from report configuration
    max_emerging_topics = report_config.get("max_emerging_topics")
    max_strong_topics = report_config.get("max_strong_topics")

    # Load signal data
    weak_signals, strong_signals = load_signal_data(
        user, model_id, reference_ts, max_emerging_topics, max_strong_topics
    )

    if (weak_signals is None or weak_signals.empty) and (
        strong_signals is None or strong_signals.empty
    ):
        logger.error(
            "No signal data available for report generation. (user: '{user}', model: '{model_id}')"
        )
        raise ValueError(
            "No signal data available for report generation. (user: '{user}', model: '{model_id}')"
        )

    # Get analysis options
    analysis_config = model_config.get("analysis_config", {})
    options = {
        "topic_evolution": analysis_config.get("topic_evolution", True),
        "evolution_scenarios": analysis_config.get("evolution_scenarios", True),
        "multifactorial_analysis": analysis_config.get("multifactorial_analysis", True),
    }

    # Create newsletter
    detailed_newsletter = create_detailed_newsletter_automated(
        weak_signals, strong_signals, model_id, reference_ts, options
    )

    # Get language from model config
    language = model_config.get("model_config", {}).get("language", "en")

    # Generate HTML report
    output_html = render_html_report(newsletter=detailed_newsletter, language=language)

    # Save to temporary file
    temp_report_path = create_temp_report(output_html)
    logger.info(f"Report generated at {temp_report_path}")

    # Send email
    send_email_automated(
        temp_path=temp_report_path,
        mail_title=report_title,
        recipients=recipients,
        model_id=model_id,
        reference_date=str(reference_ts.date()),
    )

    logger.success(
        f"Automated report generation completed for user: '{user}', model: '{model_id}'"
    )


app = typer.Typer()


@app.command("generate-report")
def generate_report_cli(
    user: str = typer.Argument(help="Identifier of the user"),
    model_id: str = typer.Argument(help="ID of the model"),
    reference_date: str = typer.Option(
        default=None,
        help="Reference date for the report (format: YYYY-MM-DD). If not provided, uses the most recent data.",
    ),
):
    """Generate and send an automated report based on model configuration"""
    generate_automated_report(user, model_id, reference_date)


if __name__ == "__main__":
    app()
