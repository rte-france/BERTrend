#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import torch

# workaround with streamlit to avoid errors Examining the path of torch.classes raised: Tried to instantiate class 'path.path’, but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

import locale

import streamlit as st

from bertrend.demos.demos_utils.i18n import (
    create_internationalization_language_selector,
    translate,
)
from bertrend.demos.demos_utils.icons import (
    NEWSLETTER_ICON,
    SAVE_ICON,
    TEMPORAL_VISUALIZATION_ICON,
    TOPIC_EXPLORATION_ICON,
    TOPIC_VISUALIZATION_ICON,
)
from bertrend.demos.demos_utils.state_utils import restore_widget_state

LAYOUT = "wide"


# Set locale for French date names
locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")


def define_pages():
    data_page = st.Page(
        page="demo_pages/training_page.py",
        title=translate("data_loading_training"),
        icon=SAVE_ICON,
    )
    topic_exploration_page = st.Page(
        page="demo_pages/explore_topics.py",
        title=translate("topic_exploration"),
        icon=TOPIC_EXPLORATION_ICON,
    )
    topic_visualization_page = st.Page(
        page="demo_pages/topic_visualizations.py",
        title=translate("topic_visualization"),
        icon=TOPIC_VISUALIZATION_ICON,
    )
    temporal_visualization_page = st.Page(
        page="demo_pages/temporal_visualizations.py",
        title=translate("temporal_visualization"),
        icon=TEMPORAL_VISUALIZATION_ICON,
    )
    newsletter_page = st.Page(
        page="demo_pages/newsletters_generation.py",
        title=translate("newsletter_generation"),
        icon=NEWSLETTER_ICON,
    )

    pg = st.navigation(
        {
            translate("topic_analysis"): [
                data_page,
                topic_exploration_page,
                topic_visualization_page,
                temporal_visualization_page,
            ],
            translate("application_example"): [newsletter_page],
        }
    )
    return pg


def main():
    pg = define_pages()
    st.set_page_config(
        page_title=translate("app_title"),
        layout=LAYOUT,
        initial_sidebar_state="expanded",
        page_icon=":part_alternation_mark:",
    )
    # Restore widget state
    restore_widget_state()

    # Add language selector to sidebar
    create_internationalization_language_selector()

    pg.run()


if __name__ == "__main__":
    main()
