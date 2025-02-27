#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import locale

import streamlit as st

from bertrend.demos.demos_utils.icons import (
    SAVE_ICON,
    TOPIC_EXPLORATION_ICON,
    TOPIC_VISUALIZATION_ICON,
    TEMPORAL_VISUALIZATION_ICON,
    NEWSLETTER_ICON,
)
from bertrend.demos.demos_utils.state_utils import restore_widget_state

TITLE = "BERTrend - Topic Analysis demo"
LAYOUT = "wide"


# Set locale for French date names
locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")


def define_pages():
    data_page = st.Page(
        page="demo_pages/training_page.py",
        title="Data loading & model training",
        icon=SAVE_ICON,
    )
    topic_exploration_page = st.Page(
        page="demo_pages/explore_topics.py",
        title="Topic exploration",
        icon=TOPIC_EXPLORATION_ICON,
    )
    topic_visualization_page = st.Page(
        page="demo_pages/topic_visualizations.py",
        title="Topic visualization",
        icon=TOPIC_VISUALIZATION_ICON,
    )
    temporal_visualization_page = st.Page(
        page="demo_pages/temporal_visualizations.py",
        title="Temporal visualization",
        icon=TEMPORAL_VISUALIZATION_ICON,
    )
    newsletter_page = st.Page(
        page="demo_pages/newsletters_generation.py",
        title="Newsletter generation",
        icon=NEWSLETTER_ICON,
    )

    pg = st.navigation(
        {
            "Topic Analysis": [
                data_page,
                topic_exploration_page,
                topic_visualization_page,
                temporal_visualization_page,
            ],
            "Application example": [newsletter_page],
        }
    )
    return pg


def main():
    pg = define_pages()
    st.set_page_config(
        page_title=TITLE,
        layout=LAYOUT,
        initial_sidebar_state="expanded",
        page_icon=":part_alternation_mark:",
    )
    # Restore widget state
    restore_widget_state()
    pg.run()


if __name__ == "__main__":
    main()
