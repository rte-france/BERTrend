#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from statistics import StatisticsError

import pandas as pd
import streamlit as st

from bertrend.demos.topic_analysis.app_utils import (
    plot_remaining_docs_repartition_over_time,
    transform_new_data,
    compute_topics_over_time,
    plot_topics_over_time,
)
from bertrend.demos.topic_analysis.state_utils import (
    restore_widget_state,
    register_widget,
    save_widget_state,
)
from bertrend.metrics.metrics import TIME_WEIGHT, TopicMetrics, TEM_x, TEM_y
from bertrend.utils.data_loading import TIMESTAMP_COLUMN

# Restore widget state
restore_widget_state()

if "tw" not in st.session_state.keys():
    st.session_state["tw"] = TIME_WEIGHT


def main():
    # Stop script if no model is trained
    if "topic_model" not in st.session_state.keys():
        st.error(
            "Train a model to explore the impact of new data on topics.", icon="🚨"
        )
        st.stop()

    if "topics_over_time" not in st.session_state.keys():
        st.error("Topics over time required.", icon="🚨")
        st.stop()

    ### TITLE ###
    st.title("Simulation of topic evolution with new data")

    timestamp_max = st.session_state["timefiltered_df"][TIMESTAMP_COLUMN].max()

    # Replace "cleaned_df" with "split_df"
    if "split_df" in st.session_state:
        st.session_state["remaining_df"] = st.session_state["split_df"].query(
            f"timestamp > '{timestamp_max}'"
        )
    else:
        st.error(
            "No split dataset available. Please ensure the dataset is properly loaded and processed.",
            icon="🚨",
        )
        st.stop()

    # Display remaining data
    st.write(f"Remaining data: {len(st.session_state['remaining_df'])} documents.")
    # Data overview
    with st.expander("Data overview"):
        freq = st.select_slider(
            "Time aggregation",
            options=(
                "1D",
                "2D",
                "1W",
                "2W",
                "1M",
                "2M",
                "1Y",
                "2Y",
            ),
            value="1M",
        )
        plot_remaining_docs_repartition_over_time(
            st.session_state["timefiltered_df"], st.session_state["remaining_df"], freq
        )

    # Select number of batches
    register_widget("new_data_batches_nb")
    st.slider(
        "Number of data batches",
        min_value=1,
        max_value=min(10, len(st.session_state["remaining_df"])),
        key="new_data_batches_nb",
        on_change=save_widget_state,
    )

    # Select time weight for topic maps evolution
    register_widget("tw")
    st.slider(
        "Time weight",
        min_value=0.0,
        max_value=0.1,
        step=0.005,
        key="tw",
        on_change=save_widget_state,
    )
    if st.button("Simulate new data", type="primary"):
        # Computes remaining_df prediction using trained topic_model
        with st.spinner("Simulating new data..."):
            st.session_state["new_topics"], _ = transform_new_data(
                st.session_state["topic_model"],
                st.session_state["remaining_df"],
                st.session_state["data_name"],
                st.session_state["widget_state"]["embedding_model_name"],
                form_parameters=st.session_state["parameters"],
                split_by_paragraphs=st.session_state["split_by_paragraphs"],
            )

    if not "new_topics" in st.session_state.keys():
        st.stop()

    with st.spinner("Computing topics over time..."):
        with st.expander("Dynamic topic modelling"):
            if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"].keys():
                st.write("## Dynamic topic modelling")

                # Parameters
                st.text_input(
                    "Topics list (format 1,12,52 or 1:20)",
                    key="dynamic_topics_list",
                    value="0:10",
                )
                st.number_input("nr_bins", min_value=1, value=10, key="nr_bins")

                # Compute topics over time using old and new data
                st.session_state["new_topics_over_time"] = compute_topics_over_time(
                    st.session_state["parameters"],
                    st.session_state["topic_model"],
                    st.session_state["timefiltered_df"],
                    nr_bins=st.session_state["nr_bins"],
                    new_df=st.session_state["remaining_df"],
                    new_nr_bins=st.session_state["new_data_batches_nb"],
                    new_topics=st.session_state["new_topics"],
                )

                # Visualize
                st.write(
                    plot_topics_over_time(
                        st.session_state["new_topics_over_time"],
                        st.session_state["dynamic_topics_list"],
                        st.session_state["topic_model"],
                        time_split=timestamp_max,
                    )
                )

    with st.expander("Topics map evolution"):
        # - plot animated topic map
        with st.spinner("Plotting topic animated map"):
            plot_animated_topic_map(
                timestamp_max,
                form_parameters=st.session_state["parameters"],
                nr_new_batches=st.session_state["new_data_batches_nb"],
            )


@st.cache_data
def plot_animated_topic_map(
    date_split: str, form_parameters: dict = None, nr_new_batches: int = 1
):
    # Topic map based on data before the introduction of new batches
    try:
        tm = TopicMetrics(
            st.session_state["topic_model"], st.session_state["topics_over_time"]
        )
        TEM_map = tm.TEM_map(st.session_state["tw"])
        TEM_map = tm.identify_signals(TEM_map, TEM_x, TEM_y)
        TEM_map["batch"] = 0
    except StatisticsError as se:
        st.warning(f"Try to change the Time Weight value: {se}", icon="⚠️")
        st.stop()

    # Use new data
    new_data = st.session_state["new_topics_over_time"][
        st.session_state["new_topics_over_time"]["Timestamp"] > date_split
    ]
    batch_results = [d for _, d in new_data.groupby(["Timestamp"])]
    current_topic_over_time = st.session_state["topics_over_time"]
    for i, batch in enumerate(batch_results):
        try:
            # Append next timestamp to current topic over time
            current_topic_over_time = pd.concat([current_topic_over_time, batch])
            # New topic metrics (that takes into account the new batch)
            topic_metrics = TopicMetrics(
                st.session_state["topic_model"], current_topic_over_time
            )
            batch_TEM_map = topic_metrics.TEM_map(st.session_state["tw"])
            batch_TEM_map = topic_metrics.identify_signals(batch_TEM_map, TEM_x, TEM_y)
            batch_TEM_map["batch"] = i + 1
            TEM_map = pd.concat([TEM_map, batch_TEM_map])
        except StatisticsError as se:
            st.warning(f"Try to change the Time Weight value: {se}", icon="⚠️")
            st.stop()

    # Plot the resulting map as an animation
    with st.spinner("Plotting topic map..."):
        st.plotly_chart(
            TopicMetrics.scatterplot_with_annotations(
                TEM_map,
                TEM_x,
                TEM_y,
                "topic",
                "topic_description",
                "Animated Topic Emergence Map (TEM)",
                TEM_x,
                TEM_y,
                animation_frame="batch",
            )
        )


###
# Write page
main()
