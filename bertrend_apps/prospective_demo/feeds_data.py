#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from bertrend import FEED_BASE_PATH
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

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Sélection de la veille",
            options=sorted(st.session_state.user_feeds.keys()),
            key="id_data",
        )

    with col2:
        if "time_window" not in st.session_state:
            st.session_state.time_window = 7
        st.slider(
            "Fenêtre temporelle (jours)",
            min_value=1,
            max_value=60,
            step=1,
            key="time_window",
        )

    display_data_info_for_feed(st.session_state.id_data)


def display_data_info_for_feed(feed_id: str):
    all_files = get_all_files_for_feed(st.session_state.user_feeds, feed_id)
    df = get_all_data(files=all_files)

    if df.empty:
        df_filtered = pd.DataFrame()
    else:
        df = df[
            [TITLE_COLUMN, URL_COLUMN, TEXT_COLUMN, TIMESTAMP_COLUMN]
        ]  # filter useful columns

        cutoff_date = datetime.datetime.now() - datetime.timedelta(
            days=st.session_state.time_window
        )
        df_filtered = df[df[TIMESTAMP_COLUMN] >= cutoff_date]

    stats = {
        "ID": feed_id,
        "# Fichiers": len(all_files),
        "Date début": df[TIMESTAMP_COLUMN].min() if not df.empty else None,
        "Date fin": df[TIMESTAMP_COLUMN].max() if not df.empty else None,
        "# Articles": len(df),
        f"# Articles ({st.session_state.time_window} derniers jours)": len(df_filtered),
    }

    st.dataframe(pd.DataFrame([stats]))

    st.write(f"#### Données des derniers {st.session_state.time_window} jours")
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("url")},
    )


@st.cache_data
def get_all_data(files: list[Path]) -> pd.DataFrame:
    """Returns the data contained in the provided files as a single DataFrame."""
    if not files:
        return pd.DataFrame()
    dfs = [load_data(Path(f)) for f in files]
    new_df = pd.concat(dfs).drop_duplicates(
        subset=["title"], keep="first", inplace=False
    )
    return new_df
