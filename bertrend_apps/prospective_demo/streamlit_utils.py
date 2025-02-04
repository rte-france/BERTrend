#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from typing import Callable

import pandas as pd
import streamlit as st


def clickable_df(
    df: pd.DataFrame, clickable_buttons: list[tuple[str | Callable, Callable, str]]
):
    """Streamlit display of a df-like rendering with additional clickable columns (buttons)."""
    cols = st.columns(len(df.columns) * [3] + len(clickable_buttons) * [1])
    for i, c in enumerate(df.columns):
        with cols[i]:
            st.write(f"**{c}**")
    for index, row in df.iterrows():
        # Create a clickable container for each row
        cols = st.columns(len(df.columns) * [3] + len(clickable_buttons) * [1])
        for i, col in enumerate(cols[: -len(clickable_buttons)]):
            with col:
                st.write(row[df.columns[i]])
        # Render the additional columns (clickable)
        for i, button in enumerate(clickable_buttons):
            with cols[len(df.columns) + i]:
                button_label = button[0](index) if callable(button[0]) else button[0]
                if st.button(button_label, key=f"button{i}_{index}", type=button[2]):
                    button[1](df.iloc[index].to_dict())
