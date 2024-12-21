#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
from bertopic import BERTopic
from pandas import Timestamp
from plotly import graph_objects as go

from bertrend import OUTPUT_PATH
from bertrend.demos.weak_signals.messages import HTML_GENERATION_FAILED_WARNING
from bertrend.demos.weak_signals.session_state_manager import SessionStateManager
from bertrend.parameters import MAX_WINDOW_SIZE, DEFAULT_WINDOW_SIZE
from bertrend.trend_analysis.visualizations import (
    create_sankey_diagram_plotly,
    plot_newly_emerged_topics,
    plot_topics_for_model,
    compute_popularity_values_and_thresholds,
    create_topic_size_evolution_figure,
    plot_topic_size_evolution,
)
from bertrend.trend_analysis.weak_signals import (
    classify_signals,
    save_signal_evolution_data,
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
) -> None:
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

    st.subheader(":grey[Noise]")
    if not noise_topics_df.empty:
        st.dataframe(
            noise_topics_df.astype(str)[columns].sort_values(
                by=["Topic", "Latest_Popularity"], ascending=[False, False]
            )
        )
    else:
        st.info(f"No noisy signals were detected at timestamp {window_end}.")

    st.subheader(":orange[Weak Signals]")
    if not weak_signal_topics_df.empty:
        st.dataframe(
            weak_signal_topics_df.astype(str)[columns].sort_values(
                by=["Latest_Popularity"], ascending=True
            )
        )
    else:
        st.info(f"No weak signals were detected at timestamp {window_end}.")

    st.subheader(":green[Strong Signals]")
    if not strong_signal_topics_df.empty:
        st.dataframe(
            strong_signal_topics_df.astype(str)[columns].sort_values(
                by=["Topic", "Latest_Popularity"], ascending=[False, False]
            )
        )
    else:
        st.info(f"No strong signals were detected at timestamp {window_end}.")


def display_popularity_evolution():
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

    # Slider to select the date
    current_date = st.slider(
        "Current date",
        min_value=min_datetime,
        max_value=max_datetime,
        step=pd.Timedelta(days=SessionStateManager.get("granularity_select")),
        format="YYYY-MM-DD",
        help="""The earliest selectable date corresponds to the earliest timestamp when topics were merged 
        (with the smallest possible value being the earliest timestamp in the provided data). 
        The latest selectable date corresponds to the most recent topic merges, which is at most equal 
        to the latest timestamp in the data minus the provided granularity.""",
        key="current_date",
    )

    granularity = SessionStateManager.get("granularity_select")

    # Compute threshold values
    window_start, window_end, all_popularity_values, q1, q3 = (
        compute_popularity_values_and_thresholds(
            bertrend.topic_sizes, window_size, granularity, current_date
        )
    )

    # Classify signals
    noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = classify_signals(
        bertrend.topic_sizes, window_start, window_end, q1, q3
    )

    # Display threshold values for noise and strong signals
    st.write(f"### Noise Threshold : {'{:.3f}'.format(q1)}")
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
    all_merge_histories_df = bertrend.all_merge_histories_df
    min_datetime = all_merge_histories_df["Timestamp"].min().to_pydatetime()
    max_datetime = all_merge_histories_df["Timestamp"].max().to_pydatetime()

    # Save Signal Evolution Data to investigate later on in a separate notebook
    start_date, end_date = st.select_slider(
        "Select date range for saving signal evolution data:",
        options=pd.date_range(
            start=min_datetime,
            end=max_datetime,
            freq=pd.Timedelta(days=SessionStateManager.get("granularity_select")),
        ),
        value=(min_datetime, max_datetime),
        format_func=lambda x: x.strftime("%Y-%m-%d"),
    )

    if st.button("Save Signal Evolution Data"):
        try:
            save_path = save_signal_evolution_data(
                all_merge_histories_df=all_merge_histories_df,
                topic_sizes=dict(bertrend.topic_sizes),
                topic_last_popularity=bertrend.topic_last_popularity,
                topic_last_update=bertrend.topic_last_update,
                window_size=SessionStateManager.get("window_size"),
                granularity=SessionStateManager.get("granularity_select"),
                start_timestamp=pd.Timestamp(start_date),
                end_timestamp=pd.Timestamp(end_date),
            )
            st.success(f"Signal evolution data saved successfully at {save_path}")
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


def display_topics_per_timestamp(topic_models: Dict[pd.Timestamp, BERTopic]) -> None:
    """
    Plot the topics discussed per source for each timestamp.

    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): A dictionary of BERTopic models, where the key is the timestamp and the value is the corresponding model.
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


def display_signal_analysis(topic_number, output_file="signal_llm.html"):
    language = SessionStateManager.get("language")
    bertrend = SessionStateManager.get("bertrend")
    all_merge_histories_df = bertrend.all_merge_histories_df

    with st.expander("Signal Interpretation", expanded=True):
        with st.spinner("Analyzing signal..."):
            summary, analysis, formatted_html = analyze_signal(
                topic_number,
                SessionStateManager.get("current_date"),
                all_merge_histories_df,
                SessionStateManager.get("granularity_select"),
                language,
            )

            # Check if the HTML file was created successfully
            output_file_path = OUTPUT_PATH / output_file
            if output_file_path.exists():
                # Read the HTML file
                with open(output_file_path, "r", encoding="utf-8") as file:
                    html_content = file.read()
                # Display the HTML content
                st.html(html_content)
            else:
                st.warning(HTML_GENERATION_FAILED_WARNING)
                # Fallback to displaying markdown if HTML generation fails
                col1, col2 = st.columns(spec=[0.5, 0.5], gap="medium")
                with col1:
                    st.markdown(summary)
                with col2:
                    st.markdown(analysis)
