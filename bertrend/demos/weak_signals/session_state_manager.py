#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import streamlit as st
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


class SessionStateManager:
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
    def get_multiple(*keys: str) -> Dict[str, Any]:
        return {key: st.session_state.get(key) for key in keys}

    @staticmethod
    def set_multiple(**kwargs: Any) -> None:
        for key, value in kwargs.items():
            st.session_state[key] = value

    @staticmethod
    def update(key: str, value: Any) -> None:
        if key in st.session_state:
            st.session_state[key] = value

    @staticmethod
    def delete(key: str) -> None:
        if key in st.session_state:
            del st.session_state[key]

    @staticmethod
    def clear() -> None:
        st.session_state.clear()

    @staticmethod
    def get_dataframe(key: str) -> Optional[pd.DataFrame]:
        df = st.session_state.get(key)
        return df if isinstance(df, pd.DataFrame) else None

    # FIXME: no usage?
    @staticmethod
    def get_model(key: str) -> Optional[Any]:
        return st.session_state.get(key)

    @staticmethod
    def get_embeddings(key: str = "embeddings") -> Optional[np.ndarray]:
        return st.session_state.get(key)
