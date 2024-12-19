#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import streamlit as st
import locale

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

from bertrend.demos.topic_analysis.app_utils import (
    plot_topics_over_time,
    compute_topics_over_time,
)

import pandas as pd

from bertrend.demos.topic_analysis.state_utils import (
    restore_widget_state,
    register_widget,
    save_widget_state,
)
from bertrend.metrics.temporal_metrics import TempTopic
from bertrend.utils.data_loading import TIMESTAMP_COLUMN, TEXT_COLUMN


def display_documents_on_click(clicked_point):
    if clicked_point:
        point = clicked_point["points"][0]
        topic_id = int(point["customdata"][0])
        timestamp = pd.to_datetime(
            point["customdata"][1], unit="D"
        )  # Convert to datetime

        topic_data = temptopic.final_df[
            (temptopic.final_df["Topic"] == topic_id)
            & (temptopic.final_df["Timestamp"] == timestamp)
        ]
        documents = topic_data["Document"].tolist()

        with st.expander(f"Documents for Topic {topic_id} at Timestamp {timestamp}"):
            for doc in documents:
                st.write(doc)


# Set locale to get French date names
locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

# Wide layout
st.set_page_config(page_title="BERTrend topic analysis", layout="wide")

# Stop script if no model is trained
if "topic_model" not in st.session_state.keys():
    st.error("Train a model to explore different temporal visualizations.", icon="🚨")
    st.stop()

# Restore widget state
restore_widget_state()


# SIDEBAR Menu
with st.sidebar:
    st.header("TEMPTopic Parameters")
    register_widget("window_size")
    window_size = st.number_input(
        "Window Size",
        min_value=2,
        value=2,
        step=1,
        key="window_size",
        on_change=save_widget_state,
    )

    register_widget("k")
    k = st.number_input(
        "Number of Nearest Embeddings (k)",
        min_value=1,
        value=1,
        step=1,
        key="k",
        on_change=save_widget_state,
    )

    register_widget("double_agg")
    double_agg = st.checkbox(
        "Double Aggregation", value=True, key="double_agg", on_change=save_widget_state
    )

    register_widget("evolution_tuning")
    evolution_tuning = st.checkbox(
        "Evolution Tuning",
        value=True,
        key="evolution_tuning",
        on_change=save_widget_state,
    )

    register_widget("global_tuning")
    global_tuning = st.checkbox(
        "Global Tuning", value=False, key="global_tuning", on_change=save_widget_state
    )

    register_widget("doc_agg")
    doc_agg_options = ["mean", "max", "min"]
    doc_agg = st.selectbox(
        "Document-level Aggregation",
        options=doc_agg_options,
        index=0,
        key="doc_agg",
        on_change=save_widget_state,
    )

    register_widget("global_agg")
    global_agg_options = ["mean", "max", "min"]
    global_agg = st.selectbox(
        "Global Aggregation",
        options=global_agg_options,
        index=1,
        key="global_agg",
        on_change=save_widget_state,
    )

    register_widget("alpha")
    alpha = st.number_input(
        "alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        key="alpha",
        on_change=save_widget_state,
    )


# Determine available time granularities based on data
min_date = st.session_state["timefiltered_df"]["timestamp"].min()
max_date = st.session_state["timefiltered_df"]["timestamp"].max()
time_diff = max_date - min_date

available_granularities = ["Day"]
if time_diff >= pd.Timedelta(weeks=1):
    available_granularities.append("Week")
if time_diff >= pd.Timedelta(days=30):
    available_granularities.append("Month")
if time_diff >= pd.Timedelta(days=365):
    available_granularities.append("Year")

# Time granularity selection
register_widget("granularity")
time_granularity = st.selectbox(
    "Select time granularity",
    [""] + available_granularities,
    key="granularity",
    on_change=save_widget_state,
)


# TEMPTopic fitting
if time_granularity != "":
    # If any of these parameters changes between two consecutive interactions, re-fit TEMPtopic
    if (
        "prev_granularity" not in st.session_state
        or st.session_state.prev_granularity != time_granularity
        or st.session_state.prev_k != k
        or st.session_state.prev_window_size != window_size
        or st.session_state.prev_double_agg != double_agg
        or st.session_state.prev_doc_agg != doc_agg
        or st.session_state.prev_global_agg != global_agg
        or st.session_state.prev_alpha != alpha
        or st.session_state.prev_evolution_tuning != evolution_tuning
        or st.session_state.prev_global_tuning != global_tuning
    ):
        # Aggregate dataframe based on selected time granularity
        df = st.session_state[
            "timefiltered_df"
        ].copy()  # Create a copy to avoid modifying the original dataframe

        # Convert 'timestamp' column to datetime if it's not already
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if time_granularity == "Day":
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        elif time_granularity == "Week":
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%W")
        elif time_granularity == "Month":
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m")
        elif time_granularity == "Year":
            df["timestamp"] = df["timestamp"].dt.strftime("%Y")

        # Group by timestamp and aggregate the text and index as lists
        aggregated_df = (
            df.groupby("timestamp")
            .agg({TEXT_COLUMN: list, "index": list})
            .reset_index()
        )

        # Extract indices from st.session_state["timefiltered_df"]["index"]
        indices = st.session_state["timefiltered_df"]["index"]

        # Extract docs using the indices
        docs = [st.session_state["split_df"][TEXT_COLUMN][i] for i in indices]

        # Create a dictionary mapping the original index to the aggregated timestamp
        index_to_timestamp = {}
        for timestamp, idx_sublist in zip(
            aggregated_df["timestamp"], aggregated_df["index"]
        ):
            for idx in idx_sublist:
                index_to_timestamp[idx] = timestamp

        # Extract the corresponding timestamps for each doc using the index_to_timestamp dictionary
        timestamps_repeated = [index_to_timestamp[idx] for idx in indices]

        # Update TempTopic with the new docs and timestamps
        temptopic = TempTopic(
            st.session_state["topic_model"],
            docs,
            st.session_state["embeddings"],
            st.session_state["token_embeddings"],
            st.session_state["token_strings"],
            timestamps_repeated,
            evolution_tuning=evolution_tuning,
            global_tuning=global_tuning,
        )

        # Display a spinner while fitting TempTopic
        with st.spinner("Fitting TempTopic..."):
            temptopic._topics_over_time()

        # Store the fitted TempTopic object and its parameters in the session state
        st.session_state.temptopic = temptopic
        st.session_state.aggregated_df = aggregated_df
        st.session_state.prev_window_size = window_size
        st.session_state.prev_k = k
        st.session_state.prev_double_agg = double_agg
        st.session_state.prev_evolution_tuning = evolution_tuning
        st.session_state.prev_global_tuning = global_tuning
        st.session_state.prev_doc_agg = doc_agg
        st.session_state.prev_global_agg = global_agg
        st.session_state.prev_granularity = time_granularity
        st.session_state.prev_alpha = alpha

    else:
        # Use the previously fitted TEMPTopic and its parameters
        alpha = st.session_state.prev_alpha
        time_granularity = st.session_state.prev_granularity
        global_agg = st.session_state.prev_global_agg
        doc_agg = st.session_state.prev_doc_agg
        global_tuning = st.session_state.prev_global_tuning
        evolution_tuning = st.session_state.prev_evolution_tuning
        double_agg = st.session_state.prev_double_agg
        k = st.session_state.prev_k
        window_size = st.session_state.prev_window_size
        aggregated_df = st.session_state.aggregated_df
        temptopic = st.session_state.temptopic


# TEMPTopic Results
if time_granularity != "":
    # Display the dataframes and visualizations using the saved TempTopic object

    with st.expander("Topic Evolution Dataframe"):
        st.dataframe(
            temptopic.final_df[
                ["Topic", "Words", "Document", "Frequency", "Timestamp"]
            ].sort_values(by=["Topic", "Timestamp"], ascending=[True, True]),
            use_container_width=True,
        )
        temptopic.final_df[
            ["Topic", "Words", "Document", "Frequency", "Timestamp"]
        ].sort_values(by=["Topic", "Timestamp"], ascending=[True, True]).to_json(
            "final_df_test.json"
        )

    with st.expander("Topic Info Dataframe"):
        st.dataframe(temptopic.topic_model.get_topic_info(), use_container_width=True)

    with st.expander("Documents per Date Dataframe"):
        st.dataframe(aggregated_df, use_container_width=True)

    with st.expander("TempTopic Visualizations"):
        topics_to_show = st.multiselect(
            "Topics to Show",
            options=list(temptopic.final_df["Topic"].unique()),
            default=None,
        )

        st.header("Temporal Stability Metrics")
        smoothing_factor = st.slider(
            "Smoothing Factor", min_value=0.0, max_value=1.0, value=0.2, step=0.1
        )

        temptopic.calculate_overall_topic_stability(
            window_size=window_size,
            k=k,
            double_agg=double_agg,
            doc_agg=doc_agg,
            global_agg=global_agg,
            alpha=alpha,
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_temporal_stability = temptopic.plot_temporal_stability_metrics(
                metric="topic_stability",
                smoothing_factor=smoothing_factor,
                topics_to_show=topics_to_show,
            )
            st.plotly_chart(fig_temporal_stability, use_container_width=True)

        with col2:
            fig_temporal_representation_stability = (
                temptopic.plot_temporal_stability_metrics(
                    metric="representation_stability",
                    smoothing_factor=smoothing_factor,
                    topics_to_show=topics_to_show,
                )
            )
            st.plotly_chart(
                fig_temporal_representation_stability, use_container_width=True
            )

        st.header("Overall Topic Stability")
        normalize_overall_stability = st.checkbox("Normalize", value=False)
        fig_overall_stability = temptopic.plot_overall_topic_stability(
            topics_to_show=topics_to_show, normalize=normalize_overall_stability
        )
        st.plotly_chart(fig_overall_stability, use_container_width=True)

        st.header("3D Topic Evolution Visualization")
        perplexity = st.number_input(
            "T-SNE Perplexity", min_value=5.0, max_value=50.0, value=30.0, step=1.0
        )
        learning_rate = st.number_input(
            "T-SNE Learning Rate",
            min_value=10.0,
            max_value=1000.0,
            value=200.0,
            step=10.0,
        )
        metric = st.selectbox("T-SNE Metric", ["cosine", "euclidean", "manhattan"])
        color_palette = st.selectbox("Color Palette", ["Plotly", "D3", "Alphabet"])

        fig_topic_evolution = temptopic.plot_topic_evolution(
            granularity=time_granularity,
            perplexity=perplexity,
            color_palette=color_palette,
            topics_to_show=topics_to_show,
        )
        st.plotly_chart(
            fig_topic_evolution, theme="streamlit", use_container_width=True
        )


with st.spinner("Computing topics over time..."):
    with st.expander("Popularity of topics over time"):
        if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"].keys():
            st.write("## Popularity of topics over time")

            # Parameters
            st.text_input(
                "Topics list (format 1,12,52 or 1:20)",
                key="dynamic_topics_list",
                value="0:10",
            )

            st.number_input("nr_bins", min_value=1, value=10, key="nr_bins")

            st.session_state["topics_over_time"] = compute_topics_over_time(
                st.session_state["parameters"],
                st.session_state["topic_model"],
                st.session_state["timefiltered_df"],
                nr_bins=st.session_state["nr_bins"],
            )

            # Visualize
            st.plotly_chart(
                plot_topics_over_time(
                    st.session_state["topics_over_time"],
                    st.session_state["dynamic_topics_list"],
                    st.session_state["topic_model"],
                ),
                use_container_width=True,
            )
