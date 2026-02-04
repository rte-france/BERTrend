#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from datetime import timedelta

import pandas as pd
import streamlit as st

from bertrend.demos.demos_utils.i18n import translate
from bertrend.demos.demos_utils.icons import ERROR_ICON, INFO_ICON
from bertrend.demos.demos_utils.state_utils import (
    SessionStateManager,
    register_multiple_widget,
    register_widget,
    restore_widget_state,
    save_widget_state,
)
from bertrend.demos.topic_analysis.app_utils import (
    compute_topics_over_time,
)
from bertrend.demos.weak_signals.visualizations_utils import PLOTLY_BUTTON_SAVE_CONFIG
from bertrend.metrics.temporal_metrics_embedding import TempTopic
from bertrend.topic_analysis.visualizations import (
    plot_overall_topic_stability,
    plot_temporal_stability_metrics,
    plot_topic_evolution,
    plot_topics_over_time,
)
from bertrend.utils.data_loading import (
    DOCUMENT_ID_COLUMN,
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
)


def check_model_trained():
    """Check if a model is trained and display an error if not."""
    if "topic_model" not in st.session_state:
        st.error(
            translate("train_model_first_error"),
            icon=ERROR_ICON,
        )
        st.stop()


def check_embedding_type():
    """Check the embedding service type. Remote embedding service currently not support these visualizations."""
    if SessionStateManager.get("embedding_service_type", "remote") != "local":
        st.error(
            translate("remote_embedding_service_type_not_supported_error"),
            icon=ERROR_ICON,
        )
        st.stop()


def _parameters_changed():
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
        register_multiple_widget(
            "window_size",
            "k",
            "alpha",
            "double_agg",
            "doc_agg",
            "global_agg",
            "evolution_tuning",
            "global_tuning",
        )
        with st.expander(translate("temptopic_parameters"), expanded=False):
            st.number_input(
                translate("window_size"),
                min_value=2,
                value=2,
                step=1,
                key="window_size",
                on_change=save_widget_state,
            )
            st.number_input(
                translate("k_nearest_embeddings"),
                min_value=1,
                value=1,
                step=1,
                key="k",
                on_change=save_widget_state,
                help=translate("k_nearest_help"),
            )
            st.number_input(
                translate("alpha_weight"),
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.01,
                key="alpha",
                help=translate("alpha_help"),
                on_change=save_widget_state,
            )
            st.checkbox(
                translate("use_double_agg"),
                value=True,
                key="double_agg",
                on_change=save_widget_state,
                help=translate("double_agg_help"),
            )
            st.selectbox(
                translate("doc_agg_method"),
                ["mean", "max"],
                key="doc_agg",
                on_change=save_widget_state,
            )
            st.selectbox(
                translate("global_agg_method"),
                ["max", "mean"],
                key="global_agg",
                on_change=save_widget_state,
            )
            st.checkbox(
                translate("use_evolution_tuning"),
                value=True,
                key="evolution_tuning",
                on_change=save_widget_state,
            )
            st.checkbox(
                translate("use_global_tuning"),
                value=False,
                key="global_tuning",
                on_change=save_widget_state,
            )


def _format_timedelta(td: timedelta) -> str:
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


def select_time_granularity(max_granularity: timedelta) -> timedelta:
    """Allow user to select custom time granularity within limits."""
    col0, col1, col2, col3, col4 = st.columns(5)

    max_days = max_granularity.days
    register_multiple_widget(
        "granularity_days", "granularity_hours", "granularity_minutes"
    )
    with col0:
        st.write(translate("select_time_granularity"))
    with col1:
        days = st.slider(
            translate("days"),
            min_value=0,
            max_value=max_days,
            value=min(1, max_days),
            key="granularity_days",
            on_change=save_widget_state,
        )
    with col2:
        hours = st.slider(
            translate("hours"),
            min_value=0,
            max_value=23,
            value=0,
            key="granularity_hours",
            on_change=save_widget_state,
        )
    with col3:
        minutes = st.slider(
            translate("minutes"),
            min_value=0,
            max_value=59,
            value=0,
            key="granularity_minutes",
            on_change=save_widget_state,
        )
    with col4:
        seconds = st.slider(
            translate("seconds"),
            min_value=0,
            max_value=59,
            value=0,
            key="granularity_seconds",
            on_change=save_widget_state,
        )

    selected_granularity = timedelta(
        days=days, hours=hours, minutes=minutes, seconds=seconds
    )
    formatted_max = _format_timedelta(max_granularity)

    st.info(
        translate("granularity_info").format(max_granularity=formatted_max),
        icon=INFO_ICON,
    )

    if (
        selected_granularity.total_seconds() == 0
        or selected_granularity > max_granularity
    ):
        st.stop()

    return pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _calculate_max_granularity(df: pd.DataFrame) -> timedelta:
    """Calculate the maximum allowed granularity based on the timestamp range."""
    max_timestamp, min_timestamp = (
        df[TIMESTAMP_COLUMN].max(),
        df[TIMESTAMP_COLUMN].min(),
    )
    time_range = max_timestamp - min_timestamp
    max_granularity = time_range / 2
    return max_granularity


def _group_timestamps(timestamps: pd.DataFrame, granularity: timedelta):
    """Group timestamps based on a custom granularity."""
    # Find the minimum timestamp
    min_timestamp = timestamps.min()

    # Calculate the difference from the minimum timestamp
    diff = timestamps - min_timestamp

    # Integer divide the difference by the granularity
    groups = diff // granularity

    # Calculate the new timestamps
    return min_timestamp + groups * granularity


def process_data_and_fit_temptopic(time_granularity: timedelta):
    """Process data and fit TempTopic with custom time granularity."""
    df = st.session_state["time_filtered_df"].copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

    # Convert time_granularity to timedelta if it's not already
    if not isinstance(time_granularity, timedelta):
        time_granularity = pd.Timedelta(time_granularity)

    # Group timestamps using the custom function
    df["grouped_timestamp"] = _group_timestamps(df[TIMESTAMP_COLUMN], time_granularity)

    aggregated_df = (
        df.groupby("grouped_timestamp")
        .agg({TEXT_COLUMN: list, DOCUMENT_ID_COLUMN: list})
        .reset_index()
    )
    indices = st.session_state["time_filtered_df"][DOCUMENT_ID_COLUMN]
    docs = [st.session_state["split_df"][TEXT_COLUMN][i] for i in indices]

    index_to_timestamp = {
        idx: timestamp
        for timestamp, idx_sublist in zip(
            aggregated_df["grouped_timestamp"], aggregated_df[DOCUMENT_ID_COLUMN]
        )
        for idx in idx_sublist
    }
    timestamps_repeated = [
        index_to_timestamp[idx].strftime("%Y-%m-%d %H:%M:%S") for idx in indices
    ]

    # Initialize and fit TempTopic
    with st.spinner(translate("fitting_temptopic")):
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
    with st.expander(translate("topic_evolution_dataframe")):
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
            width="stretch",
        )


def display_topic_info_dataframe():
    """Display the Topic Info Dataframe."""
    with st.expander(translate("topic_info_dataframe")):
        st.dataframe(
            st.session_state.temptopic.topic_model.get_topic_info(),
            width="stretch",
        )


def display_documents_per_date_dataframe():
    """Display the Documents per Date Dataframe."""
    with st.expander(translate("documents_per_date_dataframe")):
        st.dataframe(st.session_state.aggregated_df, width="stretch")


def display_temptopic_visualizations():
    """Display TempTopic Visualizations."""
    with st.expander(translate("temptopic_visualizations")):
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
            translate("topics_to_show"),
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
    st.header(translate("topic_evolution_header"))
    n_neighbors = st.slider(
        translate("umap_n_neighbors"), min_value=2, max_value=100, value=15, step=1
    )
    min_dist = st.slider(
        translate("umap_min_dist"), min_value=0.0, max_value=0.99, value=0.1, step=0.01
    )
    metric = st.selectbox(
        translate("umap_metric"), ["cosine", "euclidean", "manhattan"]
    )
    color_palette = st.selectbox(
        translate("color_palette"), ["Plotly", "D3", "Alphabet"]
    )

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
        fig_topic_evolution, config=PLOTLY_BUTTON_SAVE_CONFIG, width="stretch"
    )

    st.divider()


def display_overall_topic_stability(topics_to_show):
    """Display Overall Topic Stability."""
    st.header(translate("overall_topic_stability"))
    normalize_overall_stability = st.checkbox(translate("normalize"), value=False)
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
        width="stretch",
    )

    st.divider()


def display_temporal_stability_metrics(topics_to_show):
    """Display Temporal Stability Metrics."""
    st.header(translate("temporal_stability_metrics"))

    fig_topic_stability = plot_temporal_stability_metrics(
        st.session_state.temptopic,
        metric="topic_stability",
        topics_to_show=topics_to_show,
    )
    st.plotly_chart(
        fig_topic_stability, config=PLOTLY_BUTTON_SAVE_CONFIG, width="stretch"
    )

    fig_representation_stability = plot_temporal_stability_metrics(
        st.session_state.temptopic,
        metric="representation_stability",
        topics_to_show=topics_to_show,
    )
    st.plotly_chart(
        fig_representation_stability,
        config=PLOTLY_BUTTON_SAVE_CONFIG,
        width="stretch",
    )


def display_topics_popularity():
    """Display the popularity of topics over time."""
    with st.spinner(translate("computing_topics")):
        with st.expander(translate("popularity_of_topics")):
            if TIMESTAMP_COLUMN in st.session_state["time_filtered_df"]:
                st.write("## " + translate("popularity_of_topics"))

                # Parameters
                st.text_input(
                    translate("topics_list_format"),
                    key="dynamic_topics_list",
                    value="0:10",
                )

                st.number_input(
                    translate("nr_bins"), min_value=1, value=10, key="nr_bins"
                )

                # Compute topics over time
                st.session_state["topics_over_time"] = compute_topics_over_time(
                    st.session_state["topic_model"],
                    st.session_state["time_filtered_df"],
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
                    width="stretch",
                )


def main():
    """Main function to run the Streamlit topic_analysis."""
    # Check if model is trained
    check_model_trained()
    check_embedding_type()

    st.title(translate("temporal_visualizations"))

    # Display sidebar
    display_sidebar()

    # Calculate max granularity
    max_granularity = _calculate_max_granularity(st.session_state["time_filtered_df"])

    # Select time granularity
    time_granularity = select_time_granularity(max_granularity)

    col1, col2 = st.columns(2)
    with col1:
        # Add Apply button
        apply_button = st.button(translate("apply_granularity"), type="primary")
    with col2:
        register_widget("temptopic_visualizations")
        st.segmented_control(
            translate("show_table_results"),
            selection_mode="multi",
            key="temptopic_visualizations",
            on_change=save_widget_state,
            options=[
                translate("topic_evolution"),
                translate("topic_info"),
                translate("documents_per_date"),
            ],
        )

    if apply_button:
        if time_granularity is not None:
            st.session_state.granularity = time_granularity
            process_data_and_fit_temptopic(time_granularity)
        else:
            st.error(translate("select_valid_granularity"))

    # Display visualizations only if TempTopic has been fitted
    if "temptopic" in st.session_state:
        if translate("topic_evolution") in st.session_state["temptopic_visualizations"]:
            display_topic_evolution_dataframe()
        if translate("topic_info") in st.session_state["temptopic_visualizations"]:
            display_topic_info_dataframe()
        if (
            translate("documents_per_date")
            in st.session_state["temptopic_visualizations"]
        ):
            display_documents_per_date_dataframe()
        display_temptopic_visualizations()
    else:
        st.info(
            translate("apply_granularity_message"),
            icon=INFO_ICON,
        )


# Restore widget state
restore_widget_state()
main()
