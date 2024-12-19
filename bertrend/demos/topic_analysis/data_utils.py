#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import itertools
from typing import List

import pandas as pd
import streamlit as st
from pathlib import Path

from bertrend.demos.topic_analysis.app_utils import plot_docs_reparition_over_time
from bertrend.demos.topic_analysis.state_utils import save_widget_state
from bertrend.utils.data_loading import TEXT_COLUMN, TIMESTAMP_COLUMN


def data_overview(df: pd.DataFrame):
    with st.container(border=True):
        col1, col2 = st.columns([0.4, 0.6])
        freq = st.select_slider(
            "Time aggregation",
            options=(
                "1D",
                "2D",
                "1W",
                "2W",
                "1M",
                "2M",
                "1Y",
                "2Y",
            ),
            value="1M",
        )
        with col1:
            plot_docs_reparition_over_time(df, freq)
        with col2:
            st.dataframe(
                st.session_state["timefiltered_df"][
                    ["index", TEXT_COLUMN, TIMESTAMP_COLUMN]
                ],
                use_container_width=True,
            )


def choose_data(base_dir: Path, filters: List[str]):
    data_folders = sorted(
        set(
            f.parent
            for f in itertools.chain.from_iterable(
                [list(base_dir.glob(f"**/{filter}")) for filter in filters]
            )
        )
    )

    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = data_folders[0] if data_folders else base_dir

    data_options = ["None"] + sorted(
        [
            p.name
            for p in itertools.chain.from_iterable(
                [
                    list(st.session_state["data_folder"].glob(f"{filter}"))
                    for filter in filters
                ]
            )
        ]
    )

    if "data_name" not in st.session_state:
        st.session_state["data_name"] = data_options[0]

    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []

    folder_options = [folder.name for folder in data_folders]
    if not folder_options:
        st.warning("No data available!")
        st.stop()

    selected_folder_index = st.selectbox("Base folder", index=0, options=folder_options)
    selected_folder = data_folders[folder_options.index(selected_folder_index)]
    st.session_state["data_folder"] = selected_folder

    with st.container(border=True, height=300):
        data_files = sorted(
            itertools.chain.from_iterable(
                [list(selected_folder.glob(f"{filter}")) for filter in filters]
            ),
            key=lambda x: x.name,
        )
        for file in data_files:
            checkbox_key = f"file-{file.name}"
            if st.checkbox(file.name, key=checkbox_key):
                if file not in st.session_state["selected_files"]:
                    st.session_state["selected_files"].append(file)
            else:
                if file in st.session_state["selected_files"]:
                    st.session_state["selected_files"].remove(file)

    if not st.session_state["selected_files"]:
        st.stop()


def reset_all():
    # TODO: add here all state variables we want to reset when we change the data
    st.session_state.pop("timestamp_range", None)
    reset_topics()


def reset_data():
    # TODO: add here all state variables we want to reset when we change the data
    st.session_state.pop("timestamp_range", None)
    st.session_state.pop("data_name", None)
    reset_topics()


def reset_topics():
    # shall be called when we update data parameters (timestamp, min char, split, etc.)
    st.session_state.pop("selected_topic_number", None)
    st.session_state.pop("new_topics", None)
    st.session_state.pop("new_topics_over_time", None)
    save_widget_state()
