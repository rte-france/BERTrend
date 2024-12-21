#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import locale
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import umap

from bertrend.metrics.temporal_metrics_embedding import TempTopic
from bertrend.demos.topic_analysis.app_utils import (
    plot_topics_over_time,
    compute_topics_over_time,
)
from bertrend.demos.topic_analysis.state_utils import (
    register_widget,
    save_widget_state,
    restore_widget_state,
)
from bertrend.demos.weak_signals.visualizations_utils import PLOTLY_BUTTON_SAVE_CONFIG
from bertrend.utils.data_loading import TIMESTAMP_COLUMN, TEXT_COLUMN

# Set locale for French date names
locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

# Page configuration
st.set_page_config(page_title="BERTrend topic analysis", layout="wide")


# TempTopic output visualization functions
def plot_topic_evolution(
    temptopic,
    granularity,
    topics_to_show=None,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    color_palette="Plotly",
):
    topic_counts = temptopic.final_df.groupby("Topic")["Timestamp"].nunique()
    valid_topics = set(topic_counts[topic_counts > 1].index.tolist())
    all_topics = sorted(set(temptopic.final_df["Topic"].unique()) & set(valid_topics))
    topics_to_include = sorted(set(topics_to_show or all_topics) & set(valid_topics))

    topic_data = {}
    for topic_id in topics_to_include:
        topic_df = temptopic.final_df[temptopic.final_df["Topic"] == topic_id]

        if len(topic_df["Timestamp"].unique()) > 1:
            timestamps = pd.to_datetime(
                topic_df["Timestamp"], format="%Y-%m-%d %H:%M:%S"
            )
            topic_data[topic_id] = {
                "embeddings": topic_df["Embedding"].tolist(),
                "timestamps": timestamps,
                "words": topic_df["Words"].tolist(),
            }

    if not topic_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No topics to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    all_embeddings = np.vstack([data["embeddings"] for data in topic_data.values()])
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    all_embeddings_umap = reducer.fit_transform(all_embeddings)

    start_idx = 0
    for topic_id, data in topic_data.items():
        end_idx = start_idx + len(data["embeddings"])
        data["embeddings_umap"] = all_embeddings_umap[start_idx:end_idx]
        start_idx = end_idx

    fig = go.Figure()

    for topic_id, data in topic_data.items():
        topic_words = ", ".join(
            data["words"][0].split(", ")[:3]
        )  # Get first 3 words of the topic
        fig.add_trace(
            go.Scatter3d(
                x=data["embeddings_umap"][:, 0],
                y=data["embeddings_umap"][:, 1],
                z=data["timestamps"],
                mode="lines+markers",
                name=f"Topic {topic_id}: {topic_words}",
                text=[
                    f"Topic: {topic_id}<br>Timestamp: {t}<br>Words: {w}"
                    for t, w in zip(data["timestamps"], data["words"])
                ],
                hoverinfo="text",
                visible="legendonly",
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="Timestamp"),
        width=1000,
        height=1000,
    )

    return fig


def plot_temporal_stability_metrics(
    temptopic, metric, darkmode=True, topics_to_show=None
):
    if darkmode:
        fig = go.Figure(layout=go.Layout(template="plotly_dark"))
    else:
        fig = go.Figure()

    topic_counts = temptopic.final_df.groupby("Topic")["Timestamp"].nunique()
    valid_topics = set(topic_counts[topic_counts > 1].index.tolist())
    all_topics = sorted(set(temptopic.final_df["Topic"].unique()) & valid_topics)
    topics_to_include = sorted(set(topics_to_show or all_topics) & valid_topics)

    if metric == "topic_stability":
        df = temptopic.topic_stability_scores_df
        score_column = "Topic Stability Score"
        title = "Temporal Topic Stability"
    elif metric == "representation_stability":
        df = temptopic.representation_stability_scores_df
        score_column = "Representation Stability Score"
        title = "Temporal Representation Stability"
    else:
        raise ValueError(
            "Invalid metric. Choose 'topic_stability' or 'representation_stability'."
        )

    for topic_id in topics_to_include:
        topic_data = df[df["Topic ID"] == topic_id].sort_values(by="Start Timestamp")

        if topic_data.empty:
            continue

        topic_words = temptopic.final_df[temptopic.final_df["Topic"] == topic_id][
            "Words"
        ].iloc[0]
        topic_words = "_".join(topic_words.split(", ")[:3])

        x = topic_data["Start Timestamp"]
        y = topic_data[score_column]

        hover_text = []
        for _, row in topic_data.iterrows():
            if metric == "topic_stability":
                hover_text.append(
                    f"Topic: {topic_id}<br>Timestamp: {row['Start Timestamp']}<br>Score: {row[score_column]:.4f}"
                )
            else:
                hover_text.append(
                    f"Topic: {topic_id}<br>Timestamp: {row['Start Timestamp']}<br>Score: {row[score_column]:.4f}<br>Representation: {row['Start Representation']}"
                )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"{topic_id}_{topic_words}",
                text=hover_text,
                hoverinfo="text",
                line=dict(shape="spline", smoothing=0.9),
                visible="legendonly",
            )
        )

    if not fig.data:
        fig.add_annotation(
            text="No topics to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig.data = sorted(fig.data, key=lambda trace: int(trace.name.split("_")[0]))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title=f'{metric.replace("_", " ").capitalize()} Score',
        legend_title="Topic",
        hovermode="closest",
    )

    return fig


def plot_overall_topic_stability(
    temptopic, darkmode=True, normalize=False, topics_to_show=None
):
    if temptopic.overall_stability_df is None:
        temptopic.calculate_overall_topic_stability()

    df = temptopic.overall_stability_df

    if topics_to_show is not None and len(topics_to_show) > 0:
        df = df[df["Topic ID"].isin(topics_to_show)]

    df = df.sort_values(by="Topic ID")

    metric_column = (
        "Normalized Stability Score" if normalize else "Overall Stability Score"
    )
    df["ScoreNormalized"] = df[metric_column]
    df["Color"] = df["ScoreNormalized"].apply(
        lambda x: px.colors.diverging.RdYlGn[
            int(x * (len(px.colors.diverging.RdYlGn) - 1))
        ]
    )

    fig = go.Figure(layout=go.Layout(template="plotly_dark" if darkmode else "plotly"))

    for _, row in df.iterrows():
        topic_id = row["Topic ID"]
        metric_value = row[metric_column]
        words = temptopic.final_df[temptopic.final_df["Topic"] == topic_id][
            "Words"
        ].iloc[0]
        num_timestamps = row["Number of Timestamps"]

        fig.add_trace(
            go.Bar(
                x=[topic_id],
                y=[metric_value],
                marker_color=row["Color"],
                name=f"Topic {topic_id}",
                hovertext=f"Topic {topic_id}<br>Words: {words}<br>Score: {metric_value:.4f}<br>Timestamps: {num_timestamps}",
                hoverinfo="text",
                text=[num_timestamps],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Overall Topic Stability Scores",
        yaxis=dict(range=[0, 1]),
        yaxis_title="Overall Topic Stability Score",
        showlegend=False,
    )

    return fig


def check_model_trained():
    """Check if a model is trained and display an error if not."""
    if "topic_model" not in st.session_state:
        st.error(
            "Train a model to explore different temporal visualizations.", icon="🚨"
        )
        st.stop()


def parameters_changed():
    """Check if any of the parameters have changed."""
    params_to_check = [
        "window_size",
        "k",
        "alpha",
        "double_agg",
        "doc_agg",
        "global_agg",
        "evolution_tuning",
        "global_tuning",
        "granularity",
    ]
    return any(
        st.session_state.get(f"prev_{param}") != st.session_state.get(param)
        for param in params_to_check
    )


def display_sidebar():
    """Display the sidebar with TEMPTopic parameters."""
    with st.sidebar:
        st.header("TEMPTopic Parameters")

        register_widget("window_size")
        st.number_input(
            "Window Size",
            min_value=2,
            value=2,
            step=1,
            key="window_size",
            on_change=save_widget_state,
        )

        register_widget("k")
        st.number_input(
            "Number of Nearest Embeddings (k)",
            min_value=1,
            value=1,
            step=1,
            key="k",
            on_change=save_widget_state,
            help="The k-th nearest neighbor used for Topic Representation Stability calculation.",
        )

        register_widget("alpha")
        st.number_input(
            "Alpha (Topic vs Representation Stability Weight)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
            key="alpha",
            help="Closer to 1 gives more weight given to Topic Embedding Stability, Closer to 0 gives more weight to topic representation stability.",
            on_change=save_widget_state,
        )

        register_widget("double_agg")
        st.checkbox(
            "Use Double Aggregation",
            value=True,
            key="double_agg",
            on_change=save_widget_state,
            help="If unchecked, only Document Aggregation Method will be globally used.",
        )

        register_widget("doc_agg")
        st.selectbox(
            "Document Aggregation Method",
            ["mean", "max"],
            key="doc_agg",
            on_change=save_widget_state,
        )

        register_widget("global_agg")
        st.selectbox(
            "Global Aggregation Method",
            ["max", "mean"],
            key="global_agg",
            on_change=save_widget_state,
        )

        register_widget("evolution_tuning")
        st.checkbox(
            "Use Evolution Tuning",
            value=True,
            key="evolution_tuning",
            on_change=save_widget_state,
        )

        register_widget("global_tuning")
        st.checkbox(
            "Use Global Tuning",
            value=False,
            key="global_tuning",
            on_change=save_widget_state,
        )


def get_available_granularities(min_date, max_date):
    """Determine available time granularities based on the date range."""
    time_diff = max_date - min_date
    available_granularities = ["Day"]
    if time_diff >= pd.Timedelta(weeks=1):
        available_granularities.append("Week")
    if time_diff >= pd.Timedelta(days=30):
        available_granularities.append("Month")
    if time_diff >= pd.Timedelta(days=365):
        available_granularities.append("Year")
    return available_granularities


# def select_time_granularity(available_granularities):
#     """Allow user to select time granularity."""
#     register_widget("granularity")
#     time_granularity = st.selectbox("Select time granularity", [""] + available_granularities, key="granularity", on_change=save_widget_state)
#     if time_granularity == "":
#         st.info("Please select a time granularity to view the temporal visualizations.")
#         st.stop()
#     return time_granularity


def format_timedelta(td):
    """Format a timedelta object into a string with days, hours, minutes, and seconds."""
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if (
        seconds > 0 or not parts
    ):  # Always show seconds if no larger units or if it's the only non-zero unit
        parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
    return " ".join(parts)


def select_time_granularity(max_granularity):
    """Allow user to select custom time granularity within limits."""
    st.write("Select custom time granularity:")
    col1, col2, col3, col4 = st.columns(4)

    max_days = max_granularity.days

    with col1:
        days = st.number_input(
            "Days",
            min_value=0,
            max_value=max_days,
            value=min(1, max_days),
            key="granularity_days",
        )
    with col2:
        hours = st.number_input(
            "Hours", min_value=0, max_value=23, value=0, key="granularity_hours"
        )
    with col3:
        minutes = st.number_input(
            "Minutes", min_value=0, max_value=59, value=0, key="granularity_minutes"
        )
    with col4:
        seconds = st.number_input(
            "Seconds", min_value=0, max_value=59, value=0, key="granularity_seconds"
        )

    selected_granularity = timedelta(
        days=days, hours=hours, minutes=minutes, seconds=seconds
    )
    formatted_max = format_timedelta(max_granularity)

    st.info(
        f"Granularity must be greater than zero and less than or equal to {formatted_max}."
    )

    if (
        selected_granularity.total_seconds() == 0
        or selected_granularity > max_granularity
    ):
        st.stop()

    return pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def calculate_max_granularity(df):
    """Calculate the maximum allowed granularity based on the timestamp range."""
    max_timestamp, min_timestamp = (
        df[TIMESTAMP_COLUMN].max(),
        df[TIMESTAMP_COLUMN].min(),
    )
    time_range = max_timestamp - min_timestamp
    max_granularity = time_range / 2
    return max_granularity


def group_timestamps(timestamps, granularity):
    """
    Group timestamps based on a custom granularity.

    :param timestamps: Series of timestamps to group
    :param granularity: Timedelta object representing the granularity
    :return: Series of grouped timestamps
    """
    # Find the minimum timestamp
    min_timestamp = timestamps.min()

    # Calculate the difference from the minimum timestamp
    diff = timestamps - min_timestamp

    # Integer divide the difference by the granularity
    groups = diff // granularity

    # Calculate the new timestamps
    return min_timestamp + groups * granularity


def process_data_and_fit_temptopic(time_granularity):
    """Process data and fit TempTopic with custom time granularity."""
    df = st.session_state["timefiltered_df"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convert time_granularity to timedelta if it's not already
    if not isinstance(time_granularity, timedelta):
        time_granularity = pd.Timedelta(time_granularity)

    # Group timestamps using the custom function
    df["grouped_timestamp"] = group_timestamps(df["timestamp"], time_granularity)

    aggregated_df = (
        df.groupby("grouped_timestamp")
        .agg({TEXT_COLUMN: list, "index": list})
        .reset_index()
    )
    indices = st.session_state["timefiltered_df"]["index"]
    docs = [st.session_state["split_df"][TEXT_COLUMN][i] for i in indices]

    index_to_timestamp = {
        idx: timestamp
        for timestamp, idx_sublist in zip(
            aggregated_df["grouped_timestamp"], aggregated_df["index"]
        )
        for idx in idx_sublist
    }
    timestamps_repeated = [
        index_to_timestamp[idx].strftime("%Y-%m-%d %H:%M:%S") for idx in indices
    ]

    # Initialize and fit TempTopic
    with st.spinner("Fitting TempTopic..."):
        temptopic = TempTopic(
            st.session_state["topic_model"],
            docs,
            st.session_state["embeddings"],
            st.session_state["token_embeddings"],
            st.session_state["token_strings"],
            timestamps_repeated,
            evolution_tuning=st.session_state.evolution_tuning,
            global_tuning=st.session_state.global_tuning,
        )
        temptopic.fit(
            window_size=st.session_state.window_size,
            k=st.session_state.k,
            double_agg=st.session_state.double_agg,
            doc_agg=st.session_state.doc_agg,
            global_agg=st.session_state.global_agg,
        )

    # Store the fitted TempTopic object and current parameter values
    st.session_state.temptopic = temptopic
    st.session_state.aggregated_df = aggregated_df
    for param in [
        "window_size",
        "k",
        "alpha",
        "double_agg",
        "doc_agg",
        "global_agg",
        "evolution_tuning",
        "global_tuning",
        "granularity",
    ]:
        st.session_state[f"prev_{param}"] = st.session_state.get(param)


def display_topic_evolution_dataframe():
    """Display the Topic Evolution Dataframe."""
    with st.expander("Topic Evolution Dataframe"):
        columns_to_display = ["Topic", "Words", "Frequency", "Timestamp"]
        columns_present = [
            col
            for col in columns_to_display
            if col in st.session_state.temptopic.final_df.columns
        ]
        st.dataframe(
            st.session_state.temptopic.final_df[columns_present].sort_values(
                by=["Topic", "Timestamp"], ascending=[True, True]
            ),
            use_container_width=True,
        )


def display_topic_info_dataframe():
    """Display the Topic Info Dataframe."""
    with st.expander("Topic Info Dataframe"):
        st.dataframe(
            st.session_state.temptopic.topic_model.get_topic_info(),
            use_container_width=True,
        )


def display_documents_per_date_dataframe():
    """Display the Documents per Date Dataframe."""
    with st.expander("Documents per Date Dataframe"):
        st.dataframe(st.session_state.aggregated_df, use_container_width=True)


def display_temptopic_visualizations():
    """Display TempTopic Visualizations."""
    with st.expander("TempTopic Visualizations"):
        # Create a list of topics with their representations
        topic_options = sorted(
            [
                (
                    topic,
                    ", ".join(
                        st.session_state.temptopic.final_df[
                            st.session_state.temptopic.final_df["Topic"] == topic
                        ]["Words"]
                        .iloc[0]
                        .split(", ")[:3]
                    ),
                )
                for topic in st.session_state.temptopic.final_df["Topic"].unique()
            ]
        )

        # Create a dictionary for streamlit multiselect
        topic_dict = {f"Topic {topic}: {repr}": topic for topic, repr in topic_options}

        selected_topics = st.multiselect(
            "Topics to Show",
            options=list(topic_dict.keys()),
            format_func=lambda x: x,
            default=None,
        )

        # Convert selected topics back to topic numbers
        topics_to_show = [topic_dict[topic] for topic in selected_topics]

        display_topic_evolution(topics_to_show)
        display_overall_topic_stability(topics_to_show)
        display_temporal_stability_metrics(topics_to_show)


def display_topic_evolution(topics_to_show):
    """Display Topic Evolution in Time and Semantic Space."""
    st.header("Topic Evolution in Time and Semantic Space")
    n_neighbors = st.slider(
        "UMAP n_neighbors", min_value=2, max_value=100, value=15, step=1
    )
    min_dist = st.slider(
        "UMAP min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.01
    )
    metric = st.selectbox("UMAP Metric", ["cosine", "euclidean", "manhattan"])
    color_palette = st.selectbox("Color Palette", ["Plotly", "D3", "Alphabet"])

    fig_topic_evolution = plot_topic_evolution(
        st.session_state.temptopic,
        granularity=st.session_state.granularity,
        topics_to_show=topics_to_show,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        color_palette=color_palette,
    )
    st.plotly_chart(
        fig_topic_evolution, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True
    )

    st.divider()


def display_overall_topic_stability(topics_to_show):
    """Display Overall Topic Stability."""
    st.header("Overall Topic Stability")
    normalize_overall_stability = st.checkbox("Normalize", value=False)
    overall_stability_df = st.session_state.temptopic.calculate_overall_topic_stability(
        window_size=st.session_state.window_size,
        k=st.session_state.k,
        alpha=st.session_state.alpha,
    )
    fig_overall_stability = plot_overall_topic_stability(
        st.session_state.temptopic,
        topics_to_show=topics_to_show,
        normalize=normalize_overall_stability,
        darkmode=True,
    )
    st.plotly_chart(
        fig_overall_stability,
        config=PLOTLY_BUTTON_SAVE_CONFIG,
        use_container_width=True,
    )

    st.divider()


def display_temporal_stability_metrics(topics_to_show):
    """Display Temporal Stability Metrics."""
    st.header("Temporal Stability Metrics")

    fig_topic_stability = plot_temporal_stability_metrics(
        st.session_state.temptopic,
        metric="topic_stability",
        topics_to_show=topics_to_show,
    )
    st.plotly_chart(
        fig_topic_stability, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True
    )

    fig_representation_stability = plot_temporal_stability_metrics(
        st.session_state.temptopic,
        metric="representation_stability",
        topics_to_show=topics_to_show,
    )
    st.plotly_chart(
        fig_representation_stability,
        config=PLOTLY_BUTTON_SAVE_CONFIG,
        use_container_width=True,
    )


def display_topics_popularity():
    """Display the popularity of topics over time."""
    with st.spinner("Computing topics over time..."):
        with st.expander("Popularity of topics over time"):
            if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"]:
                st.write("## Popularity of topics over time")

                # Parameters
                st.text_input(
                    "Topics list (format 1,12,52 or 1:20)",
                    key="dynamic_topics_list",
                    value="0:10",
                )

                st.number_input("nr_bins", min_value=1, value=10, key="nr_bins")

                # Compute topics over time
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
                    config=PLOTLY_BUTTON_SAVE_CONFIG,
                    use_container_width=True,
                )


def main():
    """Main function to run the Streamlit topic_analysis."""
    # Restore widget state
    restore_widget_state()

    # Check if model is trained
    check_model_trained()

    # Display sidebar
    display_sidebar()

    # Calculate max granularity
    max_granularity = calculate_max_granularity(st.session_state["timefiltered_df"])

    # Select time granularity
    time_granularity = select_time_granularity(max_granularity)

    # Add Apply button
    apply_button = st.button("Apply Granularity and Parameters")

    if apply_button:
        if time_granularity is not None:
            st.session_state.granularity = time_granularity
            process_data_and_fit_temptopic(time_granularity)
        else:
            st.error("Please select a valid granularity before applying.")

    # Display visualizations only if TempTopic has been fitted
    if "temptopic" in st.session_state:
        display_topic_evolution_dataframe()
        display_topic_info_dataframe()
        display_documents_per_date_dataframe()
        display_temptopic_visualizations()
    else:
        st.info(
            "Please apply granularity and parameters to view the temporal visualizations."
        )


if __name__ == "__main__":
    main()

# FIXME: Popularity of topics over time visualization is based on the number of paragraphs instead of original articles, since it's the default BERTopic method
