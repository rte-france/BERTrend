#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd
import streamlit as st
from bertopic import BERTopic
from pandas import Timestamp
from plotly import graph_objects as go

from bertrend import SIGNAL_EVOLUTION_DATA_DIR
from bertrend.demos.demos_utils.icons import (
    SUCCESS_ICON,
    INFO_ICON,
    STRONG_SIGNAL_ICON,
    WEAK_SIGNAL_ICON,
    NOISE_ICON,
)
from bertrend.demos.demos_utils.state_utils import SessionStateManager
from bertrend.config.parameters import (
    MAX_WINDOW_SIZE,
    DEFAULT_WINDOW_SIZE,
    INDIVIDUAL_MODEL_TOPIC_COUNTS_FILE,
    CUMULATIVE_MERGED_TOPIC_COUNTS_FILE,
)
from bertrend.trend_analysis.visualizations import (
    create_sankey_diagram_plotly,
    plot_newly_emerged_topics,
    plot_topics_for_model,
    create_topic_size_evolution_figure,
    plot_topic_size_evolution,
)
from bertrend.trend_analysis.weak_signals import (
    analyze_signal,
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


def display_signal_categories_df(
    noise_topics_df: pd.DataFrame,
    weak_signal_topics_df: pd.DataFrame,
    strong_signal_topics_df: pd.DataFrame,
    window_end: Timestamp,
    columns=None,
    column_order=None,
    column_config=None,
):
    """Display the dataframes associated to each signal category: noise, weak signal, strong signal."""
    if columns is None:
        columns = [
            "Topic",
            "Sources",
            "Source_Diversity",
            "Representation",
            "Latest_Popularity",
            "Docs_Count",
            "Paragraphs_Count",
            "Latest_Timestamp",
            "Documents",
        ]
    if column_order is None:
        column_order = columns

    with st.expander(f":orange[{WEAK_SIGNAL_ICON} Weak Signals]", expanded=True):
        st.subheader(":orange[Weak Signals]")
        if not weak_signal_topics_df.empty:
            displayed_df = weak_signal_topics_df[columns].sort_values(
                by=["Latest_Popularity"], ascending=False
            )
            displayed_df["Documents"] = displayed_df["Documents"].astype(str)
            st.dataframe(
                displayed_df,
                column_order=column_order,
                column_config=column_config,
                hide_index=True,
            )

        else:
            st.info(
                f"No weak signals were detected at timestamp {window_end}.",
                icon=INFO_ICON,
            )

    with st.expander(f":green[{STRONG_SIGNAL_ICON} Strong Signals]", expanded=True):
        st.subheader(":green[Strong Signals]")
        if not strong_signal_topics_df.empty:
            displayed_df = strong_signal_topics_df[columns].sort_values(
                by=["Latest_Popularity"], ascending=False
            )
            displayed_df["Documents"] = displayed_df["Documents"].astype(str)
            st.dataframe(
                displayed_df,
                column_order=column_order,
                column_config=column_config,
                hide_index=True,
            )
        else:
            st.info(
                f"No strong signals were detected at timestamp {window_end}.",
                icon=INFO_ICON,
            )

    with st.expander(f":grey[{NOISE_ICON} Noise]", expanded=True):
        st.subheader(":grey[Noise]")
        if not noise_topics_df.empty:
            displayed_df = noise_topics_df[columns].sort_values(
                by=["Latest_Popularity"], ascending=False
            )
            displayed_df["Documents"] = displayed_df["Documents"].astype(str)
            st.dataframe(
                displayed_df,
                column_order=column_order,
                column_config=column_config,
                hide_index=True,
            )
        else:
            st.info(
                f"No noisy signals were detected at timestamp {window_end}.",
                icon=INFO_ICON,
            )


def display_popularity_evolution():
    """Display the popularity evolution diagram."""
    window_size = st.number_input(
        "Retrospective Period (days)",
        min_value=1,
        max_value=MAX_WINDOW_SIZE,
        value=DEFAULT_WINDOW_SIZE,
        key="window_size",
    )

    bertrend = SessionStateManager.get("bertrend")

    all_merge_histories_df = bertrend.all_merge_histories_df
    min_datetime = all_merge_histories_df["Timestamp"].min().to_pydatetime()
    max_datetime = all_merge_histories_df["Timestamp"].max().to_pydatetime()

    # Get granularity
    granularity = st.session_state["granularity"]

    # Slider to select the date
    current_date = st.slider(
        "Current date",
        min_value=min_datetime,
        max_value=max_datetime,
        step=pd.Timedelta(days=granularity),
        format="YYYY-MM-DD",
        help="""The earliest selectable date corresponds to the earliest timestamp when topics were merged 
        (with the smallest possible value being the earliest timestamp in the provided data). 
        The latest selectable date corresponds to the most recent topic merges, which is at most equal 
        to the latest timestamp in the data minus the provided granularity.""",
        key="current_date",
    )

    # Compute threshold values and classify signals
    window_start, window_end, all_popularity_values, q1, q3 = (
        bertrend._compute_popularity_values_and_thresholds(window_size, current_date)
    )

    # Classify signals
    noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = (
        bertrend._classify_signals(window_start, window_end, q1, q3)
    )

    # Display threshold values for noise and strong signals
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"### Noise Threshold : {'{:.3f}'.format(q1)}")
    with col2:
        st.write(f"### Strong Signal Threshold : {'{:.3f}'.format(q3)}")

    # Plot popularity evolution with thresholds
    fig = plot_topic_size_evolution(
        create_topic_size_evolution_figure(bertrend.topic_sizes),
        current_date,
        window_start,
        window_end,
        all_popularity_values,
        q1,
        q3,
    )
    st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)

    # Display DataFrames for each category noise, weak signals, strong signals
    display_signal_categories_df(
        noise_topics_df, weak_signal_topics_df, strong_signal_topics_df, window_end
    )


def save_signal_evolution():
    """Save Signal Evolution Data to investigate later on in a separate notebook"""
    bertrend = SessionStateManager.get("bertrend")
    granularity = SessionStateManager.get("granularity")
    all_merge_histories_df = bertrend.all_merge_histories_df
    min_datetime = all_merge_histories_df["Timestamp"].min().to_pydatetime()
    max_datetime = all_merge_histories_df["Timestamp"].max().to_pydatetime()

    # Save Signal Evolution Data to investigate later on in a separate notebook
    start_date, end_date = st.select_slider(
        "Select date range for saving signal evolution data:",
        options=pd.date_range(
            start=min_datetime,
            end=max_datetime,
            freq=pd.Timedelta(days=granularity),
        ),
        value=(min_datetime, max_datetime),
        format_func=lambda x: x.strftime("%Y-%m-%d"),
    )

    if st.button("Save Signal Evolution Data"):
        try:
            save_path = bertrend.save_signal_evolution_data(
                window_size=SessionStateManager.get("window_size"),
                start_timestamp=pd.Timestamp(start_date),
                end_timestamp=pd.Timestamp(end_date),
            )
            st.success(
                f"Signal evolution data saved successfully at {save_path}",
                icon=SUCCESS_ICON,
            )
        except Exception as e:
            st.error(f"Error encountered while saving signal evolution data: {e}")


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


def display_topics_per_timestamp(topic_models: dict[pd.Timestamp, BERTopic]) -> None:
    """
    Plot the topics discussed per source for each timestamp.

    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): A dictionary of BERTopic models, where the key is the timestamp
        and the value is the corresponding model.
    """
    with st.expander("Explore topic models"):
        model_periods = sorted(topic_models.keys())
        selected_model_period = st.select_slider(
            "Select Model", options=model_periods, key="model_slider"
        )
        selected_model = topic_models[selected_model_period]

        fig = plot_topics_for_model(selected_model)

        st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)

        st.dataframe(
            selected_model.doc_info_df[
                ["Paragraph", "document_id", "Topic", "Representation", "source"]
            ],
            use_container_width=True,
        )
        st.dataframe(selected_model.topic_info_df, use_container_width=True)


def display_signal_analysis(topic_number: int):
    """Display a LLM-based analyis of a specific topic."""
    bertrend = SessionStateManager.get("bertrend")

    st.subheader("Signal Interpretation")
    with st.spinner("Analyzing signal..."):
        summary, analysis, formatted_html = analyze_signal(
            bertrend,
            topic_number,
            SessionStateManager.get("current_date"),
        )

        # Display the HTML content
        st.html(formatted_html)


def retrieve_topic_counts(topic_models: dict[pd.Timestamp, BERTopic]) -> None:
    individual_model_topic_counts = [
        (timestamp, model.topic_info_df["Topic"].max() + 1)
        for timestamp, model in topic_models.items()
    ]
    df_individual_models = pd.DataFrame(
        individual_model_topic_counts,
        columns=["timestamp", "num_topics"],
    )

    # Number of topics per cumulative merged model
    cumulative_merged_topic_counts = SessionStateManager.get(
        "merge_df_size_over_time", []
    )
    df_cumulative_merged = pd.DataFrame(
        cumulative_merged_topic_counts,
        columns=["timestamp", "num_topics"],
    )

    # Convert to JSON
    json_individual_models = df_individual_models.to_json(
        orient="records", date_format="iso", indent=4
    )
    json_cumulative_merged = df_cumulative_merged.to_json(
        orient="records", date_format="iso", indent=4
    )

    # Save individual model topic counts
    json_file_path = (
        SIGNAL_EVOLUTION_DATA_DIR
        / f"retrospective_{SessionStateManager.get('window_size')}_days"
    )
    json_file_path.mkdir(parents=True, exist_ok=True)

    (json_file_path / INDIVIDUAL_MODEL_TOPIC_COUNTS_FILE).write_text(
        json_individual_models
    )

    # Save cumulative merged model topic counts
    (json_file_path / CUMULATIVE_MERGED_TOPIC_COUNTS_FILE).write_text(
        json_cumulative_merged
    )
    st.success(
        f"Topic counts for individual and cumulative merged models saved to {json_file_path}",
        icon=SUCCESS_ICON,
    )
