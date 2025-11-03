#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st

from bertrend_apps.prospective_demo.i18n import translate
from bertrend.demos.demos_utils.icons import CYCLE_ICON
from bertrend.utils.data_loading import (
    load_data,
    TIMESTAMP_COLUMN,
    TITLE_COLUMN,
    URL_COLUMN,
    TEXT_COLUMN,
)
from bertrend_apps.prospective_demo.feeds_common import get_all_files_for_feed


def display_data_status():
    if not st.session_state.user_feeds:
        return

    col1, col2, col3 = st.columns([45, 45, 10])
    with col1:
        st.selectbox(
            translate("monitoring_selection_label"),
            options=sorted(st.session_state.user_feeds.keys()),
            key="id_data",
        )

    with col2:
        if "data_time_window" not in st.session_state:
            st.session_state.data_time_window = 7
        st.slider(
            translate("time_window_label"),
            min_value=1,
            max_value=60,
            step=1,
            key="data_time_window",
        )

    with col3:
        clicked = st.button(CYCLE_ICON + " " + translate("check_data_button_label"))

    if clicked:
        display_data_info_for_feed(st.session_state.id_data)


def display_data_info_for_feed(feed_id: str):
    all_files = get_all_files_for_feed(st.session_state.user_feeds, feed_id)
    df = get_all_data_parallel(files=all_files)

    if df.empty:
        df_filtered = pd.DataFrame()
    else:
        df = df[
            [TITLE_COLUMN, URL_COLUMN, TEXT_COLUMN, TIMESTAMP_COLUMN]
        ]  # filter useful columns

        cutoff_date = datetime.datetime.now() - datetime.timedelta(
            days=st.session_state.data_time_window
        )
        df_filtered = df[df[TIMESTAMP_COLUMN] >= cutoff_date]

    stats = {
        translate("stats_id_label"): feed_id,
        translate("stats_files_count_label"): len(all_files),
        translate("stats_start_date_label"): (
            df[TIMESTAMP_COLUMN].min() if not df.empty else None
        ),
        translate("stats_end_date_label"): (
            df[TIMESTAMP_COLUMN].max() if not df.empty else None
        ),
        translate("stats_articles_count_label"): len(df),
        translate("stats_recent_articles_count_label").format(
            days=st.session_state.data_time_window
        ): len(df_filtered),
    }

    st.dataframe(pd.DataFrame([stats]))

    st.write(
        f"#### {translate('recent_data_title').format(days=st.session_state.data_time_window)}"
    )
    st.dataframe(
        df_filtered,
        width="stretch",
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("url")},
    )


@st.cache_data
def get_all_data(files: list[Path]) -> pd.DataFrame:
    """Returns the data contained in the provided files as a single DataFrame."""
    if not files:
        return pd.DataFrame()
    dfs = [load_data(Path(f)) for f in files]
    # Filter out None/empty DataFrames before concatenation
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    # Optimize concat: ignore_index speeds up concatenation, copy=False avoids unnecessary copies
    new_df = pd.concat(dfs, ignore_index=True, copy=False)
    # Drop duplicates - keep='first' ensures we keep the first occurrence
    new_df = new_df.drop_duplicates(subset=["title"], keep="first")
    return new_df


@st.cache_data
def get_all_data_parallel(files: list[Path]) -> pd.DataFrame:
    """Returns the data contained in the provided files as a single DataFrame, using parallel loading."""
    if not files:
        return pd.DataFrame()

    TOTAL_CORES = os.cpu_count()
    if TOTAL_CORES is None:
        # Fallback to a safe number if the core count is undetermined
        TOTAL_CORES = 4
    # Set MAX_WORKERS to system cores - 1, ensuring a minimum of 1 worker
    MAX_WORKERS = max(1, TOTAL_CORES - 1)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit load_data tasks for all files
        # executor.map is efficient for simple function calls across an iterable
        dfs = list(executor.map(load_data, files))

    # Filter out None/empty DataFrames
    dfs = [df for df in dfs if df is not None and not df.empty]

    if not dfs:
        return pd.DataFrame()

    # Concatenate and process
    new_df = pd.concat(dfs, ignore_index=True, copy=False)
    new_df = new_df.drop_duplicates(subset=["title"], keep="first")
    return new_df
