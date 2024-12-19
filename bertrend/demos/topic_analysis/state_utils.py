#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import streamlit as st

STATE_KEYS = "state_keys"
WIDGET_STATE = "widget_state"

# Set of util functions to manage the state of widgets in streamlit
# - at the beginning of each page: add restore_widget_state() to restore the last selected values
# - each time a widget is used, call *before* register_widget with the *key* of the widget as argument
# and use the call_back of the widget (for example on_change) to call save_widget_state


def register_widget(key):
    if STATE_KEYS not in st.session_state.keys():
        st.session_state[STATE_KEYS] = []
    if key not in st.session_state[STATE_KEYS]:
        st.session_state[STATE_KEYS].append(key)


def save_widget_state():
    if STATE_KEYS in st.session_state.keys():
        st.session_state[WIDGET_STATE] = {
            key: st.session_state[key]
            for key in st.session_state[STATE_KEYS]
            if key in st.session_state
        }


def restore_widget_state():
    if WIDGET_STATE in st.session_state.keys():
        for k, v in st.session_state[WIDGET_STATE].items():
            st.session_state[k] = v
