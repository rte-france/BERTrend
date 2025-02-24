#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pandas as pd
import streamlit as st

from bertrend.demos.demos_utils.icons import WARNING_ICON
from bertrend.demos.weak_signals.visualizations_utils import (
    display_signal_categories_df,
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
            "Titre", pinned=True, width="large"
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
        display_signal_categories_df(
            dfs_topics[NOISE],
            dfs_topics[WEAK_SIGNALS],
            dfs_topics[STRONG_SIGNALS],
            reference_ts,
            columns=columns,
            column_config=column_config,
        )

    with col2:
        st.info(
            "TODO <Place disponible pour d'autres infos...>"
            "En particulier, rajouter la courbe d'évolution des sujets"
            "en se limitant à ceux présentés dans les tableaux weak/strong/noise"
        )
        explore_topic_sources(dfs_topics)


@st.fragment
def explore_topic_sources(dfs_topics):
    st.write("**Exploration des sources par sujet**")
    selected_signal_type = st.pills(
        "Type de signal",
        label_visibility="hidden",
        options=["Sujets émergents", "Sujets forts"],
        selection_mode="single",
        default="Sujets émergents",
    )
    if selected_signal_type == "Sujets forts":
        selected_df = dfs_topics.get(STRONG_SIGNALS)
    else:
        selected_df = dfs_topics.get(WEAK_SIGNALS)
    if selected_df is None or selected_df.empty:
        st.warning(f"{WARNING_ICON} Pas de données")
    else:
        selected_df = selected_df.sort_values(by=["Latest_Popularity"], ascending=False)
        options = selected_df["Topic"].tolist()
        topic_id = st.selectbox(
            index=None,
            label="Sélection du sujet",
            label_visibility="hidden",
            options=options,
            format_func=lambda x: f"Sujet {x}: "
            + selected_df[selected_df["Topic"] == x][LLM_TOPIC_TITLE_COLUMN].values[0],
        )
        if topic_id is None:
            return
        if selected_signal_type == "Sujets forts":
            color = "green"
        else:
            color = "orange"
        row = selected_df[selected_df["Topic"] == topic_id]
        display_topic_links(
            title=f":{color}[**{row[LLM_TOPIC_TITLE_COLUMN].values[0]}**]",
            desc=row[LLM_TOPIC_DESCRIPTION_COLUMN].values[0],
            df=list(selected_df.query(f"Topic == {topic_id}")[URLS_COLUMN])[0],
        )


@st.dialog("Exploration des sources", width="large")
def display_topic_links(title: str, desc: str, df: pd.DataFrame):
    st.subheader(title)
    st.write(desc)
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "value": st.column_config.LinkColumn("Articles de référence"),
        },
    )
