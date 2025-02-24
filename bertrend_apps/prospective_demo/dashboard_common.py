#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import uuid
from typing import Any

import pandas as pd
import streamlit as st

from bertrend.demos.demos_utils.icons import WARNING_ICON
from bertrend_apps.prospective_demo import NOISE, WEAK_SIGNALS, STRONG_SIGNALS
from bertrend_apps.prospective_demo.models_info import get_models_info

COLS_RATIO_ID_TS = [2 / 7, 5 / 7]


@st.cache_data
def get_df_topics(model_interpretation_path=None) -> dict[str, pd.DataFrame]:
    dfs_topics = {}
    for df_id in [NOISE, WEAK_SIGNALS, STRONG_SIGNALS]:
        df_path = model_interpretation_path / f"{df_id}.parquet"
        dfs_topics[df_id] = (
            pd.read_parquet(df_path) if df_path.exists() else pd.DataFrame()
        )
    return dfs_topics


def update_key(key: str, new_value: Any):
    st.session_state[key] = new_value
    # reset ts value in order to avoid errors if a previous ts value is not available for the new key value


def update_key_and_ts(key: str, new_value: Any):
    update_key(key, new_value)
    # reset ts value in order to avoid errors if a previous ts value is not available for the new key value
    if "reference_ts" in st.session_state:
        del st.session_state["reference_ts"]


def choose_id_and_ts():
    col1, col2 = st.columns(COLS_RATIO_ID_TS)
    with col1:
        options = sorted(st.session_state.user_feeds.keys())
        if "model_id" not in st.session_state:
            st.session_state.model_id = options[0]
        model_id_key = uuid.uuid4()
        model_id = st.selectbox(
            "Sélection de la veille",
            options=options,
            index=options.index(st.session_state.model_id),
            key=model_id_key,  # to avoid pb of unicity if displayed on several places
            on_change=lambda: update_key_and_ts(
                "model_id", st.session_state[model_id_key]
            ),
        )
    with col2:
        list_models = get_models_info(model_id)
        if not list_models:
            st.warning(f"{WARNING_ICON} Pas de modèle disponible")
            st.stop()
        elif len(list_models) < 2:
            st.warning(
                f"{WARNING_ICON} 2 modèles minimum pour analyser les tendances !"
            )
            st.stop()
        if "reference_ts" not in st.session_state:
            st.session_state.reference_ts = list_models[-1]
        ts_key = uuid.uuid4()
        st.select_slider(
            "Date d'analyse",
            options=list_models,
            value=st.session_state.reference_ts,
            format_func=lambda ts: ts.strftime("%d/%m/%Y"),
            help="Sélection de la date d'analyse parmi celles disponibles",
            key=ts_key,  # to avoid pb of unicity if displayed on several places
            on_change=lambda: update_key("reference_ts", st.session_state[ts_key]),
        )
