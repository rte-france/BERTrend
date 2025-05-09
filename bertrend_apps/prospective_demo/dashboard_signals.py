#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pandas as pd
import streamlit as st

from bertrend.demos.demos_utils.icons import (
    WARNING_ICON,
    WEAK_SIGNAL_ICON,
    STRONG_SIGNAL_ICON,
    NOISE_ICON,
)
from bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    NOISE,
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    get_model_interpretation_path,
    LLM_TOPIC_TITLE_COLUMN,
    URLS_COLUMN,
)
from bertrend_apps.prospective_demo.dashboard_common import (
    choose_id_and_ts,
    get_df_topics,
)
from bertrend_apps.prospective_demo.i18n import translate

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

    dfs_topics = get_df_topics(model_interpretation_path)

    col1, col2 = st.columns(COLS_RATIO)
    with col1:
        # Display dataframes for weak_signals, strong, etc
        display_translated_signal_categories(
            dfs_topics[NOISE],
            dfs_topics[WEAK_SIGNALS],
            dfs_topics[STRONG_SIGNALS],
            reference_ts,
            columns=columns,
            column_config=column_config,
        )

    with col2:
        st.info(translate("todo_message"))
        explore_topic_sources(dfs_topics)


@st.fragment
def explore_topic_sources(dfs_topics):
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
        options = selected_df["Topic"].tolist()
        topic_id = st.selectbox(
            index=None,
            label=translate("topic_selection"),
            label_visibility="hidden",
            options=options,
            format_func=lambda x: f"{translate('topic')} {x}: "
            + selected_df[selected_df["Topic"] == x][LLM_TOPIC_TITLE_COLUMN].values[0],
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


def display_translated_signal_categories(
    noise_topics_df: pd.DataFrame,
    weak_signal_topics_df: pd.DataFrame,
    strong_signal_topics_df: pd.DataFrame,
    window_end: pd.Timestamp,
    columns=None,
    column_order=None,
    column_config=None,
):
    """Wrapper around display_signal_categories_df that uses translated text."""
    # Weak Signals
    with st.expander(
        f":orange[{WEAK_SIGNAL_ICON} {translate('weak_signals')}]", expanded=True
    ):
        st.subheader(f":orange[{translate('weak_signals')}]")
        if not weak_signal_topics_df.empty:
            displayed_df = weak_signal_topics_df[columns].sort_values(
                by=["Latest_Popularity"], ascending=False
            )
            displayed_df["Documents"] = displayed_df["Documents"].astype(str)
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
            displayed_df = strong_signal_topics_df[columns].sort_values(
                by=["Latest_Popularity"], ascending=False
            )
            displayed_df["Documents"] = displayed_df["Documents"].astype(str)
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
            displayed_df = noise_topics_df[columns].sort_values(
                by=["Latest_Popularity"], ascending=False
            )
            displayed_df["Documents"] = displayed_df["Documents"].astype(str)
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


@st.dialog(translate("explore_sources"), width="large")
def display_topic_links(title: str, desc: str, df: pd.DataFrame):
    st.subheader(title)
    st.write(desc)
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "value": st.column_config.LinkColumn(translate("reference_articles")),
        },
    )
