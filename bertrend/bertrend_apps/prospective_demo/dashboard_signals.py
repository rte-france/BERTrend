#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pandas as pd
import streamlit as st

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    NOISE,
    STRONG_SIGNALS,
    URLS_COLUMN,
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
    PROMOTED_TOPIC,
    TopicFeedback,
    apply_topic_feedback,
    get_topic_feedback_icon,
    load_topic_feedback,
    order_topic_ids,
)
from bertrend.demos.demos_utils.icons import (
    NOISE_ICON,
    STRONG_SIGNAL_ICON,
    WARNING_ICON,
    WEAK_SIGNAL_ICON,
)

COLS_RATIO = [4 / 7, 3 / 7]


def signal_analysis():
    # ID and timestamp selection
    choose_id_and_ts()
    model_id = st.session_state.model_id
    reference_ts = st.session_state.reference_ts

    model_interpretation_path = get_model_interpretation_path(
        user_name=st.session_state.username,
        model_id=model_id,
        reference_ts=reference_ts,
    )

    # TODO: decide which column to display
    columns = [
        "Topic",
        LLM_TOPIC_TITLE_COLUMN,
        LLM_TOPIC_DESCRIPTION_COLUMN,
        # "Representation",
        URLS_COLUMN,
        "Latest_Popularity",
        "Docs_Count",
        # "Paragraphs_Count",
        "Documents",
        "Sources",
        "Source_Diversity",
        "Latest_Timestamp",
    ]
    column_config = {
        "Topic": st.column_config.NumberColumn(
            "Topic",
            pinned=True,
        ),
        LLM_TOPIC_TITLE_COLUMN: st.column_config.TextColumn(
            translate("title"), pinned=True, width="large"
        ),
        "Latest_Popularity": st.column_config.ProgressColumn(
            format="%i",
            max_value=50,
        ),
        "Source_Diversity": st.column_config.ProgressColumn(
            format="%i",
            max_value=50,
        ),
        "Latest_Timestamp": st.column_config.DateColumn(
            format="DD/MM/YYYY",
        ),
        URLS_COLUMN: st.column_config.LinkColumn(),
    }

    feedback = load_topic_feedback(
        get_user_models_path(st.session_state.username, model_id)
    )
    raw_topics = get_df_topics(model_interpretation_path)
    # Feedback applied once for source exploration (selectbox options / ordering).
    # Table display applies it separately after a popularity sort in
    # _prepare_topics_for_display — pass raw frames there to avoid a redundant pass.
    dfs_topics = {
        category: apply_topic_feedback(topics, feedback)
        for category, topics in raw_topics.items()
    }

    col1, col2 = st.columns(COLS_RATIO)
    with col1:
        # Display dataframes for weak_signals, strong, etc
        display_translated_signal_categories(
            raw_topics[NOISE],
            raw_topics[WEAK_SIGNALS],
            raw_topics[STRONG_SIGNALS],
            reference_ts,
            columns=columns,
            column_config=column_config,
            feedback=feedback,
        )

    with col2:
        explore_topic_sources(dfs_topics, feedback)


@st.fragment
def explore_topic_sources(
    dfs_topics: dict[str, pd.DataFrame], feedback: dict[int, TopicFeedback]
):
    st.write(f"**{translate('explore_sources_by_topic')}**")
    selected_signal_type = st.pills(
        translate("signal_type"),
        label_visibility="hidden",
        options=[translate("emerging_topics"), translate("strong_topics")],
        selection_mode="single",
        default=translate("emerging_topics"),
    )
    if selected_signal_type == translate("strong_topics"):
        selected_df = dfs_topics.get(STRONG_SIGNALS)
    else:
        selected_df = dfs_topics.get(WEAK_SIGNALS)
    if selected_df is None or selected_df.empty:
        st.warning(f"{WARNING_ICON} {translate('no_data')}")
    else:
        selected_df = selected_df.sort_values(by=["Latest_Popularity"], ascending=False)
        options = order_topic_ids(selected_df["Topic"].tolist(), feedback)
        topic_id = st.selectbox(
            index=None,
            label=translate("topic_selection"),
            label_visibility="hidden",
            options=options,
            format_func=lambda topic: _format_source_topic_label(
                topic,
                selected_signal_type,
                selected_df,
                feedback,
            ),
        )
        if topic_id is None:
            return
        if selected_signal_type == translate("strong_topics"):
            color = "green"
        else:
            color = "orange"
        row = selected_df[selected_df["Topic"] == topic_id]
        display_topic_links(
            title=f":{color}[**{row[LLM_TOPIC_TITLE_COLUMN].values[0]}**]",
            desc=row[LLM_TOPIC_DESCRIPTION_COLUMN].values[0],
            df=list(selected_df.query(f"Topic == {topic_id}")[URLS_COLUMN])[0],
        )


def _format_source_topic_label(
    topic_id: int,
    selected_signal_type: str,
    selected_df: pd.DataFrame,
    feedback: dict[int, TopicFeedback],
) -> str:
    is_emerging = selected_signal_type == translate("emerging_topics")
    category_icon = "📈" if is_emerging else "🌟"
    category_label = translate("emerging_topic" if is_emerging else "strong_topic")
    title = selected_df.loc[
        selected_df["Topic"] == topic_id, LLM_TOPIC_TITLE_COLUMN
    ].iloc[0]
    if pd.isna(title) or not title:
        title = translate("untitled_topic")
    feedback_icon = get_topic_feedback_icon(topic_id, feedback)
    prefix = f"{feedback_icon} " if feedback_icon else ""
    return f"{prefix}{category_icon} [{category_label} {topic_id}] {title}"


def display_translated_signal_categories(
    noise_topics_df: pd.DataFrame,
    weak_signal_topics_df: pd.DataFrame,
    strong_signal_topics_df: pd.DataFrame,
    window_end: pd.Timestamp,
    columns=None,
    column_order=None,
    column_config=None,
    feedback: dict[int, TopicFeedback] | None = None,
):
    """Wrapper around display_signal_categories_df that uses translated text."""
    feedback = feedback or {}
    # Weak Signals
    with st.expander(
        f":orange[{WEAK_SIGNAL_ICON} {translate('weak_signals')}]", expanded=True
    ):
        st.subheader(f":orange[{translate('weak_signals')}]")
        if not weak_signal_topics_df.empty:
            displayed_df = _prepare_topics_for_display(
                weak_signal_topics_df, columns, feedback
            )
            st.dataframe(
                displayed_df,
                column_order=column_order if column_order else columns,
                column_config=column_config,
                hide_index=True,
            )
        else:
            st.info(
                translate("no_weak_signals").format(timestamp=window_end),
                icon=WARNING_ICON,
            )

    # Strong Signals
    with st.expander(
        f":green[{STRONG_SIGNAL_ICON} {translate('strong_signals')}]", expanded=True
    ):
        st.subheader(f":green[{translate('strong_signals')}]")
        if not strong_signal_topics_df.empty:
            displayed_df = _prepare_topics_for_display(
                strong_signal_topics_df, columns, feedback
            )
            st.dataframe(
                displayed_df,
                column_order=column_order if column_order else columns,
                column_config=column_config,
                hide_index=True,
            )
        else:
            st.info(
                translate("no_strong_signals").format(timestamp=window_end),
                icon=WARNING_ICON,
            )

    # Noise
    with st.expander(f":grey[{NOISE_ICON} {translate('noise')}]", expanded=True):
        st.subheader(f":grey[{translate('noise')}]")
        if not noise_topics_df.empty:
            displayed_df = _prepare_topics_for_display(
                noise_topics_df, columns, feedback
            )
            st.dataframe(
                displayed_df,
                column_order=column_order if column_order else columns,
                column_config=column_config,
                hide_index=True,
            )
        else:
            st.info(
                translate("no_noise_signals").format(timestamp=window_end),
                icon=WARNING_ICON,
            )


def _prepare_topics_for_display(
    topics: pd.DataFrame,
    columns: list[str],
    feedback: dict[int, TopicFeedback],
) -> pd.DataFrame:
    """Prepare a signal table while keeping promoted topics visible first.

    Callers should pass raw (unfiltered) topic frames. Popularity is sorted first,
    then apply_topic_feedback re-orders so promoted topics still float to the top
    while relative popularity order is preserved within each feedback group.
    """
    displayed_topics = topics[columns].sort_values(
        by=["Latest_Popularity"], ascending=False
    )
    displayed_topics = apply_topic_feedback(displayed_topics, feedback)
    promoted_topic_ids = {
        topic_id for topic_id, status in feedback.items() if status == PROMOTED_TOPIC
    }
    promoted = displayed_topics["Topic"].isin(promoted_topic_ids)
    displayed_topics.loc[promoted, LLM_TOPIC_TITLE_COLUMN] = (
        "⭐ "
        + displayed_topics.loc[promoted, LLM_TOPIC_TITLE_COLUMN]
        .fillna(translate("untitled_topic"))
        .astype(str)
    )
    displayed_topics["Documents"] = displayed_topics["Documents"].astype(str)
    return displayed_topics


@st.dialog(translate("explore_sources"), width="large")
def display_topic_links(title: str, desc: str, df: pd.DataFrame):
    st.subheader(title)
    st.write(desc)
    st.dataframe(
        df,
        width="stretch",
        column_config={
            "value": st.column_config.LinkColumn(translate("reference_articles")),
        },
    )
