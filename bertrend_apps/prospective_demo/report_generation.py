#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import pandas as pd
import streamlit as st
from google.auth.exceptions import RefreshError
from loguru import logger

from bertrend_apps.prospective_demo.i18n import translate
from bertrend.demos.demos_utils.icons import (
    NEWSLETTER_ICON,
    TOPIC_ICON,
    ERROR_ICON,
    DOWNLOAD_ICON,
    EMAIL_ICON,
)
from bertrend.demos.streamlit_components.input_with_pills_component import (
    input_with_pills,
)
from bertrend.llm_utils.newsletter_model import (
    STRONG_TOPIC_TYPE,
    WEAK_TOPIC_TYPE,
    Article,
)
from bertrend.trend_analysis.data_structure import TopicSummaryList, SignalAnalysis
from bertrend_apps.common.mail_utils import get_credentials, send_email
from bertrend_apps.prospective_demo import (
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    URLS_COLUMN,
)
from bertrend_apps.prospective_demo.dashboard_common import choose_id_and_ts
from bertrend_apps.prospective_demo.data_model import DetailedNewsletter, TopicOverTime
from bertrend_apps.prospective_demo.report_generation_utils import (
    render_html_report,
    create_temp_report,
    MAXIMUM_NUMBER_OF_ARTICLES,
)
from bertrend_apps.prospective_demo.utils import is_valid_email


def reporting():
    choose_id_and_ts()
    selected_weak_topics_df, selected_strong_topics_df = choose_topics()
    configure_export(selected_weak_topics_df, selected_strong_topics_df)


def choose_topics():
    st.subheader(translate(TOPIC_ICON + " " + translate("step_1_title")))
    model_id = st.session_state.model_id
    dfs_interpretation = st.session_state.signal_interpretations
    if model_id not in dfs_interpretation:
        st.error(f"{ERROR_ICON} {translate('no_data')}")
        st.stop()
    cols = st.columns(2)
    with cols[0]:
        st.write(f"#### :orange[{translate('emerging_topics')}]")
        if WEAK_SIGNALS not in dfs_interpretation[model_id]:
            st.error(f"{ERROR_ICON} {translate('no_data')}")
            filtered_weak_signals = None
        else:
            df_w = dfs_interpretation[model_id][WEAK_SIGNALS]
            weak_topics_list = choose_from_df(df_w)
            filtered_weak_signals = df_w[df_w["Topic"].isin(weak_topics_list)]

    with cols[1]:
        st.write(f"#### :green[{translate('strong_topics')}]")
        if STRONG_SIGNALS not in dfs_interpretation[model_id]:
            st.error(f"{ERROR_ICON} {translate('no_data')}")
            filtered_strong_signals = None
        else:
            df_w = dfs_interpretation[model_id][STRONG_SIGNALS]
            strong_topics_list = choose_from_df(df_w)
            filtered_strong_signals = df_w[df_w["Topic"].isin(strong_topics_list)]

    return filtered_weak_signals, filtered_strong_signals


def choose_from_df(df: pd.DataFrame):
    df["A retenir"] = True
    df["Sujet"] = df[LLM_TOPIC_TITLE_COLUMN]
    df["Description"] = df[LLM_TOPIC_DESCRIPTION_COLUMN]
    columns = ["Topic", "A retenir", "Sujet", "Description"]
    edited_df = st.data_editor(df[columns], num_rows="dynamic", column_order=columns)
    selection = edited_df[edited_df["A retenir"] == True]["Topic"].tolist()
    return selection


def configure_export(weak_signals: pd.DataFrame, strong_signals: pd.DataFrame):
    st.subheader(translate(NEWSLETTER_ICON + " " + translate("step_2_title")))
    model_id = st.session_state.model_id
    topic_evolution = st.checkbox(
        translate("topic_evolution"),
        value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
            "topic_evolution"
        ],
    )
    evolution_scenarios = st.checkbox(
        translate("evolution_scenarios"),
        value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
            "evolution_scenarios"
        ],
    )
    multifactorial_analysis = st.checkbox(
        translate("multifactorial_analysis"),
        value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
            "multifactorial_analysis"
        ],
    )
    options = {
        "topic_evolution": topic_evolution,
        "evolution_scenarios": evolution_scenarios,
        "multifactorial_analysis": multifactorial_analysis,
    }
    st.button(
        translate("generate_button_label"),
        type="primary",
        on_click=lambda: generate_report(weak_signals, strong_signals, options),
    )


def create_detailed_newsletter(
    weak_signals: pd.DataFrame, strong_signals: pd.DataFrame, options: dict = None
) -> DetailedNewsletter:
    model_id = st.session_state.model_id

    # Create the DetailedNewsletter object
    detailed_newsletter = DetailedNewsletter(
        title=model_id,
        reference_period=st.session_state.reference_ts.date(),
        topics=[],
    )

    # Iterate over topics
    for df, topic_type in zip(
        [weak_signals, strong_signals], [WEAK_TOPIC_TYPE, STRONG_TOPIC_TYPE]
    ):
        # Iterate over the filtered DataFrame rows
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            # For each row, iterate over each column
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
                topic_analysis=SignalAnalysis.model_validate_json(
                    row["analysis"],
                ),
            )

            if options is not None and not options["topic_evolution"]:
                topic.topic_evolution = None
            if options is not None and not options["evolution_scenarios"]:
                topic.topic_analysis.evolution_scenario = None
            if options is not None and not options["multifactorial_analysis"]:
                topic.topic_analysis.potential_implications = None
                topic.topic_analysis.topic_interconnexions = None
                topic.topic_analysis.drivers_inhibitors = None

            detailed_newsletter.topics.append(topic)

    return detailed_newsletter


@st.dialog(translate("report_preview_title"), width="large")
def generate_report(
    weak_signals: pd.DataFrame, strong_signals: pd.DataFrame, options: dict = None
):
    model_id = st.session_state.model_id

    detailed_newsletter: DetailedNewsletter = create_detailed_newsletter(
        weak_signals, strong_signals, options
    )
    language = st.session_state.model_analysis_cfg[model_id]["model_config"]["language"]
    # Generate the HTML
    output_html = render_html_report(
        newsletter=detailed_newsletter,
        language=language,
    )

    # Render HTML
    with st.container(height=475):
        st.html(output_html)

    # Save report to temp file
    temp_report_path = create_temp_report(output_html)  # Create the file in temp dir

    col1, col2, _ = st.columns([3, 3, 7])
    with col1:
        download(temp_report_path, model_id)
    with col2:
        download_json(detailed_newsletter, model_id)
    recipients = input_with_pills(
        label=translate("email_recipients"),
        key="emails",
        placeholder=translate("email_recipients"),
        validate_fn=is_valid_email,
        label_visibility="hidden",
    )
    email(
        temp_report_path,
        mail_title=f"{translate('report_mail_title')} {model_id}",
        recipients=recipients,
    )


def download_json(detailed_newsletter: DetailedNewsletter, model_id: str):
    st.download_button(
        label=translate("download_json_button_label"),
        type="primary",
        data=detailed_newsletter.model_dump_json(),
        file_name=f"{st.session_state.reference_ts.date()}_{model_id}.json",
        mime="application/json",
        icon=DOWNLOAD_ICON,
    )


def download(temp_path: Path, model_id: str):
    with open(temp_path, "r", encoding="utf-8") as file:
        st.download_button(
            label=translate("download_button_label"),
            type="primary",
            data=file.read(),
            file_name=f"{st.session_state.reference_ts.date()}_{model_id}.html",
            mime="text/html",
            icon=DOWNLOAD_ICON,
        )


def email(temp_path: Path, mail_title: str, recipients: list[str]) -> None:
    if st.button(f"{EMAIL_ICON} {translate('send_button_label')}", type="primary"):
        if not recipients:
            return
        if not all(is_valid_email(email) for email in recipients):
            st.error(f"{ERROR_ICON} {translate('invalid_email')}")
            return

        try:
            # Send the newsletter by email
            try:
                if recipients:
                    credentials = get_credentials()
                    with open(temp_path, "r") as file:
                        # Read the entire contents of the file into a string
                        # content = file.read()
                        st.info(translate("email_being_sent"))
                        send_email(
                            credentials=credentials,
                            subject=mail_title,
                            recipients=recipients,
                            content=temp_path,
                            file_name=f"{st.session_state.reference_ts.date()}_{st.session_state.model_id}.html",
                        )
            except RefreshError as re:
                logger.error(
                    f"Problem with token for email, please regenerate it: {re}"
                )

            st.success(translate("email_sent_successfully"))

        except Exception as e:
            st.error(f"{translate('email_error_message')}: {e}")
