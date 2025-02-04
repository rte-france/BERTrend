#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st


@st.fragment()
def dashboard_analysis():
    """Dashboard to analyze information monitoring results"""

    selected_id = st.selectbox(
        "Sélection de la veille", options=sorted(st.session_state.user_feeds.keys())
    )
    st.write(selected_id)
