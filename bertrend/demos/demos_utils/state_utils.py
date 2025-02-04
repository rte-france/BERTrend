#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from typing import Any

import numpy as np
import pandas as pd

import streamlit as st

STATE_KEYS = "state_keys"
WIDGET_STATE = "widget_state"

# Set of util functions to manage the state of widgets in streamlit
# - at the beginning of each page: add restore_widget_state() to restore the last selected values
# - each time a widget is used, call *before* register_widget with the *key* of the widget as argument
# and use the call_back of the widget (for example on_change) to call save_widget_state


def register_widget(key: str):
    """Keeps track of widget state in case of multi-pages Streamlit app. It shall be called with the key of
    the widget before the widget creation."""
    if STATE_KEYS not in st.session_state.keys():
        st.session_state[STATE_KEYS] = []
    if key not in st.session_state[STATE_KEYS]:
        st.session_state[STATE_KEYS].append(key)


def reset_widget_state(key: str):
    """Removes the widget state from session state."""
    if WIDGET_STATE in st.session_state.keys():
        st.session_state[WIDGET_STATE].pop(key)


def register_multiple_widget(*keys: str):
    """Keeps track of widget state in case of multi-pages Streamlit app. It shall be called with the key of
    the widget before the widget creation."""
    for key in keys:
        register_widget(key)


def save_widget_state():
    """(Callback) Function to save the widget state in case of multi-pages Streamlit app.
    It shall be used as the call_back of the widget (for example on_change)."""
    if STATE_KEYS in st.session_state.keys():
        st.session_state[WIDGET_STATE] = {
            key: st.session_state[key]
            for key in st.session_state[STATE_KEYS]
            if key in st.session_state
        }


def restore_widget_state():
    """Function to restore widget states in case of multi-pages Streamlit app.
    It shall be called at the start of each page."""
    if WIDGET_STATE in st.session_state.keys():
        for k, v in st.session_state[WIDGET_STATE].items():
            st.session_state[k] = v


class SessionStateManager:
    """Class to ease the management of Streamlit session state"""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        st.session_state[key] = value

    @staticmethod
    def get_or_set(key: str, default: Any) -> Any:
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    @staticmethod
    def get_multiple(*keys: str) -> dict[str, Any]:
        return {key: st.session_state.get(key) for key in keys}

    @staticmethod
    def set_multiple(**kwargs: Any) -> None:
        for key, value in kwargs.items():
            st.session_state[key] = value

    @staticmethod
    def clear() -> None:
        st.session_state.clear()

    @staticmethod
    def get_dataframe(key: str) -> pd.DataFrame | None:
        df = st.session_state.get(key)
        return df if isinstance(df, pd.DataFrame) else None

    @staticmethod
    def get_embeddings(key: str = "embeddings") -> np.ndarray | None:
        return st.session_state.get(key)
