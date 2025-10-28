#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from importlib.metadata import version

import torch

# workaround with streamlit to avoid errors Examining the path of torch.classes raised: Tried to instantiate class 'path.path’, but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

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
    TIMELINE_ICON,
)
from bertrend.demos.demos_utils.state_utils import SessionStateManager
from bertrend_apps.prospective_demo.authentication import check_password
from bertrend_apps.prospective_demo.dashboard_analysis import dashboard_analysis
from bertrend_apps.prospective_demo.dashboard_comparative import dashboard_comparative
from bertrend_apps.prospective_demo.feeds_config import configure_information_sources
from bertrend_apps.prospective_demo.feeds_data import display_data_status
from bertrend.demos.demos_utils.i18n import (
    set_internationalization_language,
    create_internationalization_language_selector,
)
from bertrend_apps.prospective_demo.i18n import translate
from bertrend_apps.prospective_demo.models_info import models_monitoring
from bertrend_apps.prospective_demo.report_generation import reporting
from bertrend_apps.prospective_demo.dashboard_signals import signal_analysis


# UI Settings
LAYOUT: Literal["centered", "wide"] = "wide"

AUTHENTIFICATION = True
# AUTHENTIFICATION = False


def display_version_number():
    # Display version at the bottom (using st.markdown and CSS for fixed footer)
    footer = f"""
    <style>
    .footer {{
        position: fixed;
        right: 10px;
        bottom: 10px;
        color: #666;
        font-size: 0.8em;
        opacity: 0.6;
        user-select: none;
        pointer-events: none;
        background: none;
    }}
    </style>
    <div class="footer">
        Version: {version('bertrend')}
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)


def main():
    """Main page"""
    set_internationalization_language("fr")
    page_title = translate("app_title")
    st.set_page_config(
        page_title=page_title,
        layout=LAYOUT,
        initial_sidebar_state="expanded" if is_admin_mode() else "collapsed",
        page_icon=":part_alternation_mark:",
    )

    display_version_number()

    st.title(":part_alternation_mark: " + page_title)

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
        st.header(SETTINGS_ICON + " " + translate("settings_and_controls"))
        create_internationalization_language_selector()

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            NEWSLETTER_ICON + " " + translate("tab_monitoring"),
            MODELS_ICON + " " + translate("tab_models"),
            TREND_ICON + " " + translate("tab_trends"),
            ANALYSIS_ICON + " " + translate("tab_analysis"),
            NEWSLETTER_ICON + " " + translate("tab_reports"),
            TIMELINE_ICON + " " + translate("tab_comparative"),
        ]
    )

    with tab1:
        with st.expander(
            translate("data_flow_config"), expanded=True, icon=SETTINGS_ICON
        ):
            configure_information_sources()

        with st.expander(
            translate("data_collection_status"),
            expanded=False,
            icon=SERVER_STORAGE_ICON,
        ):
            display_data_status()
    with tab2:
        with st.expander(
            translate("model_status_by_monitoring"), expanded=True, icon=MODELS_ICON
        ):
            models_monitoring()

    with tab3:
        signal_analysis()

    with tab4:
        dashboard_analysis()

    with tab5:
        reporting()

    with tab6:
        dashboard_comparative()


if __name__ == "__main__":
    main()
