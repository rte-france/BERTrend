#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import pandas as pd
import streamlit as st

from bertrend.demos.demos_utils.icons import ERROR_ICON
from bertrend.trend_analysis.data_structure import SignalAnalysis, TopicSummaryList
from bertrend.trend_analysis.prompts import fill_html_template
from bertrend_apps.prospective_demo import (
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    get_model_interpretation_path,
)
from bertrend_apps.prospective_demo.dashboard_common import (
    choose_id_and_ts,
    get_df_topics,
)
from bertrend_apps.prospective_demo.i18n import translate, get_current_language


@st.fragment()
def dashboard_analysis():
    """Dashboard to analyze information monitoring results"""
    st.session_state.signal_interpretations = {}
    choose_id_and_ts()

    # LLM-based interpretation
    model_id = st.session_state.model_id
    reference_ts = st.session_state.reference_ts

    model_interpretation_path = get_model_interpretation_path(
        user_name=st.session_state.username,
        model_id=model_id,
        reference_ts=reference_ts,
    )

    # Detailed analysis
    st.subheader(translate("detailed_analysis_by_topic"))
    dfs_topics = get_df_topics(model_interpretation_path)
    display_detailed_analysis(model_id, model_interpretation_path, dfs_topics)


@st.fragment()
def display_detailed_analysis(
    model_id: str, model_interpretation_path: Path, dfs_topics: dict[str, pd.DataFrame]
):
    # Retrieve previously computed interpretation
    interpretations = {}
    for df_id, df in dfs_topics.items():
        interpretation_file_path = (
            model_interpretation_path / f"{df_id}_interpretation.jsonl"
        )
        interpretation_df = pd.read_json(interpretation_file_path, lines=True)
        if not df.empty and not interpretation_df.empty:
            interpretations[df_id] = (
                pd.merge(
                    interpretation_df,
                    df,
                    how="left",
                    left_on="topic",
                    right_on="Topic",
                )
                if interpretation_file_path.exists()
                else {}
            )

    signal_topics = {WEAK_SIGNALS: [], STRONG_SIGNALS: []}
    if WEAK_SIGNALS in interpretations:
        signal_topics[WEAK_SIGNALS] = list(interpretations[WEAK_SIGNALS]["topic"])
    if STRONG_SIGNALS in interpretations:
        signal_topics[STRONG_SIGNALS] = list(interpretations[STRONG_SIGNALS]["topic"])
    signal_list = signal_topics[WEAK_SIGNALS] + signal_topics[STRONG_SIGNALS]
    selected_signal = st.selectbox(
        label=translate("topic_selection"),
        label_visibility="hidden",
        options=signal_list,
        format_func=lambda signal_id: f"[{translate('topic')} {translate('emerging_topic') if signal_id in signal_topics[WEAK_SIGNALS] else translate('strong_topic')} "
        f"{signal_id}]: {get_row(signal_id, interpretations[WEAK_SIGNALS] if signal_id in signal_topics[WEAK_SIGNALS] else interpretations[STRONG_SIGNALS])[LLM_TOPIC_TITLE_COLUMN]}",
    )
    # Summary of the topic
    desc = get_row(
        selected_signal,
        (
            interpretations[WEAK_SIGNALS]
            if selected_signal in signal_topics[WEAK_SIGNALS]
            else (
                interpretations[STRONG_SIGNALS]
                if selected_signal in signal_topics[STRONG_SIGNALS]
                else None
            )
        ),
    )
    if desc is None:
        st.error(f"{ERROR_ICON} {translate('nothing_to_display')}")
        return
    if selected_signal in list(signal_topics[WEAK_SIGNALS]):
        color = "orange"
    else:
        color = "green"
    st.subheader(f":{color}[**{desc[LLM_TOPIC_TITLE_COLUMN]}**]")
    st.write(desc[LLM_TOPIC_DESCRIPTION_COLUMN])

    # Detailed description (HTML formatted)
    summaries: TopicSummaryList = TopicSummaryList.model_validate_json(desc["summary"])
    signal_analysis: SignalAnalysis = SignalAnalysis.model_validate_json(
        desc["analysis"]
    )
    # Use current language for HTML template
    lang = get_current_language()
    formatted_html = fill_html_template(summaries, signal_analysis, lang)
    st.html(formatted_html)

    st.session_state.signal_interpretations[model_id] = interpretations


def get_row(signal_id: int, df: pd.DataFrame) -> str | None:
    if df is None:
        return None
    filtered_df = df[df["topic"] == signal_id]
    if not filtered_df.empty:
        return filtered_df.iloc[0]  # Return the Series (row)
    else:
        st.warning(f"No data found for signal ID: {signal_id}")
