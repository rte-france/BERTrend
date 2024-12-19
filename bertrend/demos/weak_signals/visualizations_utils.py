#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd
import streamlit as st
from plotly import graph_objects as go

from bertrend.trend_analysis.visualizations import (
    create_sankey_diagram_plotly,
    plot_newly_emerged_topics,
)

PLOTLY_BUTTON_SAVE_CONFIG = {
    "toImageButtonOptions": {
        "format": "svg",
        # 'height': 500,
        # 'width': 1500,
        "scale": 1,
    }
}


def display_sankey_diagram(all_merge_histories_df: pd.DataFrame) -> None:
    """
    Create a Sankey diagram to visualize the topic merging process.

    Args:
        all_merge_histories_df (pd.DataFrame): The DataFrame containing all merge histories.

    Returns:
        go.Figure: The Plotly figure representing the Sankey diagram.
    """

    with st.expander("Topic Merging Process", expanded=False):
        # Create search box and slider using Streamlit
        search_term = st.text_input("Search topics by keyword:")
        max_pairs = st.slider(
            "Max number of topic pairs to display",
            min_value=1,
            max_value=1000,
            value=20,
        )

        # Create the Sankey diagram
        sankey_diagram = create_sankey_diagram_plotly(
            all_merge_histories_df, search_term, max_pairs
        )

        # Display the diagram using Streamlit in an expander
        st.plotly_chart(
            sankey_diagram, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True
        )


def display_newly_emerged_topics(all_new_topics_df: pd.DataFrame) -> None:
    """
    Display the newly emerged topics over time (dataframe and figure).

    Args:
        all_new_topics_df (pd.DataFrame): The DataFrame containing information about newly emerged topics.
    """
    fig_new_topics = plot_newly_emerged_topics(all_new_topics_df)

    with st.expander("Newly Emerged Topics", expanded=False):
        st.dataframe(
            all_new_topics_df[
                [
                    "Topic",
                    "Count",
                    "Document_Count",
                    "Representation",
                    "Documents",
                    "Timestamp",
                ]
            ].sort_values(by=["Timestamp", "Document_Count"], ascending=[True, False])
        )
        st.plotly_chart(
            fig_new_topics, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True
        )
