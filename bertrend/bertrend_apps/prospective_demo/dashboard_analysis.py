#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import pandas as pd
import streamlit as st

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    STRONG_SIGNALS,
    WEAK_SIGNALS,
    get_model_interpretation_path,
    get_user_models_path,
)
from bertrend.bertrend_apps.prospective_demo.dashboard_common import (
    choose_id_and_ts,
    get_df_topics,
)
from bertrend.bertrend_apps.prospective_demo.i18n import translate
from bertrend.bertrend_apps.prospective_demo.topic_feedback import (
    HIDDEN_TOPIC,
    PROMOTED_TOPIC,
    TopicFeedback,
    get_topic_feedback_icon,
    load_topic_feedback,
    order_topic_ids,
    set_topic_feedback,
)
from bertrend.demos.demos_utils.icons import ERROR_ICON
from bertrend.trend_analysis.data_structure import SignalAnalysis, TopicSummaryList
from bertrend.trend_analysis.prompts import fill_html_template

NO_TOPIC_FEEDBACK = "none"
TOPIC_FEEDBACK_SAVED_MESSAGE_KEY = "_topic_feedback_saved"


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
    model_path = get_user_models_path(st.session_state.username, model_id)

    # Detailed analysis
    st.subheader(translate("detailed_analysis_by_topic"))
    if st.session_state.pop(TOPIC_FEEDBACK_SAVED_MESSAGE_KEY, False):
        st.success(translate("topic_feedback_saved"))
    dfs_topics = get_df_topics(model_interpretation_path)
    display_detailed_analysis(
        model_id, model_interpretation_path, model_path, dfs_topics
    )


@st.fragment()
def display_detailed_analysis(
    model_id: str,
    model_interpretation_path: Path,
    model_path: Path,
    dfs_topics: dict[str, pd.DataFrame],
):
    feedback = load_topic_feedback(model_path)

    # Retrieve previously computed interpretation
    interpretations = {}
    for df_id, df in dfs_topics.items():
        interpretation_file_path = (
            model_interpretation_path / f"{df_id}_interpretation.jsonl"
        )
        if not interpretation_file_path.exists():
            continue

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
    signal_list = order_topic_ids(
        signal_topics[WEAK_SIGNALS] + signal_topics[STRONG_SIGNALS],
        feedback,
        include_hidden=True,
    )
    selected_signal = st.selectbox(
        label=translate("topic_selection"),
        label_visibility="hidden",
        options=signal_list,
        format_func=lambda signal_id: _format_signal_label(
            signal_id,
            signal_topics,
            interpretations,
            feedback,
        ),
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
        st.error(translate("nothing_to_display"), icon=ERROR_ICON)
        return
    if selected_signal in list(signal_topics[WEAK_SIGNALS]):
        color = "orange"
    else:
        color = "green"
    st.subheader(f":{color}[**{desc[LLM_TOPIC_TITLE_COLUMN]}**]")
    st.write(desc[LLM_TOPIC_DESCRIPTION_COLUMN])
    _display_topic_feedback_control(
        model_id,
        model_path,
        selected_signal,
        feedback,
    )

    # Detailed description (HTML formatted)
    # Handle nan values for summary and analysis
    if pd.notna(desc["summary"]) and pd.notna(desc["analysis"]):
        summaries: TopicSummaryList = TopicSummaryList.model_validate_json(
            desc["summary"]
        )
        signal_analysis: SignalAnalysis = SignalAnalysis.model_validate_json(
            desc["analysis"]
        )
        # Use current language for HTML template
        lang = st.session_state.model_analysis_cfg[model_id]["model_config"]["language"]
        formatted_html = fill_html_template(summaries, signal_analysis, lang)
        st.html(formatted_html)
    else:
        st.warning(translate("no_analysis_available"))

    st.session_state.signal_interpretations[model_id] = interpretations


def _format_signal_label(
    signal_id: int,
    signal_topics: dict[str, list[int]],
    interpretations: dict[str, pd.DataFrame],
    feedback: dict[int, TopicFeedback],
) -> str:
    is_weak_signal = signal_id in signal_topics[WEAK_SIGNALS]
    category = WEAK_SIGNALS if is_weak_signal else STRONG_SIGNALS
    category_icon = "📈" if is_weak_signal else "🌟"
    category_label = translate("emerging_topic" if is_weak_signal else "strong_topic")
    row = get_row(signal_id, interpretations.get(category))
    title = (
        row[LLM_TOPIC_TITLE_COLUMN] if row is not None else translate("untitled_topic")
    )
    feedback_icon = get_topic_feedback_icon(signal_id, feedback)
    prefix = f"{feedback_icon} " if feedback_icon else ""
    return f"{prefix}{category_icon} [{category_label} {signal_id}] {title}"


def _display_topic_feedback_control(
    model_id: str,
    model_path: Path,
    topic_id: int,
    feedback: dict[int, TopicFeedback],
) -> None:
    current_feedback = feedback.get(int(topic_id))
    selected_feedback = st.segmented_control(
        translate("topic_feedback"),
        options=[NO_TOPIC_FEEDBACK, PROMOTED_TOPIC, HIDDEN_TOPIC],
        default=current_feedback or NO_TOPIC_FEEDBACK,
        format_func=lambda status: translate(f"topic_feedback_{status}"),
        help=translate("topic_feedback_help"),
        selection_mode="single",
        key=f"topic_feedback_{model_id}_{topic_id}",
    )
    new_feedback = (
        None if selected_feedback in {None, NO_TOPIC_FEEDBACK} else selected_feedback
    )
    if new_feedback != current_feedback:
        set_topic_feedback(model_path, topic_id, new_feedback)
        st.session_state[TOPIC_FEEDBACK_SAVED_MESSAGE_KEY] = True
        st.rerun(scope="app")


def get_row(signal_id: int, df: pd.DataFrame) -> str | None:
    if df is None:
        return None
    filtered_df = df[df["topic"] == signal_id]
    if not filtered_df.empty:
        return filtered_df.iloc[0]  # Return the Series (row)
    else:
        st.warning(translate("no_data_for_signal").format(signal_id=signal_id))
        return None
