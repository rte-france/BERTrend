#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from typing import Literal

import streamlit as st

from bertrend.demos.demos_utils import is_admin_mode
from bertrend.demos.demos_utils.icons import (
    SETTINGS_ICON,
    ANALYSIS_ICON,
    NEWSLETTER_ICON,
    SERVER_STORAGE_ICON,
    TREND_ICON,
    MODELS_ICON,
)
from bertrend.demos.demos_utils.state_utils import SessionStateManager
from bertrend_apps.prospective_demo.authentication import check_password
from bertrend_apps.prospective_demo.dashboard_analysis import dashboard_analysis
from bertrend_apps.prospective_demo.feeds_config import configure_information_sources
from bertrend_apps.prospective_demo.feeds_data import display_data_status
from bertrend_apps.prospective_demo.models_info import models_monitoring
from bertrend_apps.prospective_demo.report_generation import reporting
from bertrend_apps.prospective_demo.dashboard_signals import signal_analysis

# UI Settings
# PAGE_TITLE = "BERTrend - Prospective Analysis demo"
PAGE_TITLE = "BERTrend - Démo Veille & Analyse"
LAYOUT: Literal["centered", "wide"] = "wide"

# TODO: reactivate password
# AUTHENTIFICATION = True
AUTHENTIFICATION = False


def main():
    """Main page"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=LAYOUT,
        initial_sidebar_state="expanded" if is_admin_mode() else "collapsed",
        page_icon=":part_alternation_mark:",
    )

    st.title(":part_alternation_mark: " + PAGE_TITLE)

    if AUTHENTIFICATION:
        username = check_password()
        if not username:
            st.stop()
        else:
            SessionStateManager.set("username", username)
    else:
        SessionStateManager.get_or_set(
            "username", "nemo"
        )  # if username is not set or authentication deactivated

    # Sidebar
    with st.sidebar:
        st.header(SETTINGS_ICON + " Settings and Controls")

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            NEWSLETTER_ICON + " Veilles",
            MODELS_ICON + " Modèles",
            TREND_ICON + " Tendances",
            ANALYSIS_ICON + " Analyses",
            NEWSLETTER_ICON + " Génération de rapports",
        ]
    )

    with tab1:
        with st.expander(
            "Configuration des flux de données", expanded=True, icon=SETTINGS_ICON
        ):
            configure_information_sources()

        with st.expander(
            "Etat de collecte des données", expanded=False, icon=SERVER_STORAGE_ICON
        ):
            display_data_status()
    with tab2:
        with st.expander(
            "Statut des modèles par veille", expanded=True, icon=MODELS_ICON
        ):
            models_monitoring()

    with tab3:
        signal_analysis()

    with tab4:
        dashboard_analysis()

    with tab5:
        reporting()


if __name__ == "__main__":
    main()
