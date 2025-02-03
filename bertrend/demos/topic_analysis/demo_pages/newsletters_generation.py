#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import inspect

import pandas as pd
import streamlit as st
from pathlib import Path

from bertrend.demos.demos_utils.icons import ERROR_ICON
from bertrend.demos.demos_utils.state_utils import (
    register_widget,
    save_widget_state,
    restore_widget_state,
    register_multiple_widget,
)
from bertrend.demos.topic_analysis.messages import TRAIN_MODEL_FIRST_ERROR
from bertrend.services.summary.abstractive_summarizer import AbstractiveSummarizer
from bertrend.services.summary.chatgpt_summarizer import GPTSummarizer
from bertrend.services.summary.extractive_summarizer import (
    ExtractiveSummarizer,
    EnhancedExtractiveSummarizer,
)
from bertrend.llm_utils.newsletter_features import generate_newsletter, md2html


# Define summarizer options
SUMMARIZER_OPTIONS_MAPPER = {
    "GPTSummarizer": GPTSummarizer,
    "AbstractiveSummarizer": AbstractiveSummarizer,
    "ExtractiveSummarizer": ExtractiveSummarizer,
    "EnhancedExtractiveSummarizer": EnhancedExtractiveSummarizer,
}


def generate_newsletter_wrapper(
    df: pd.DataFrame, df_split: pd.DataFrame
) -> tuple[str, str, str]:
    """Wrapper function to generate newsletter based on user settings."""
    top_n_topics = (
        None
        if st.session_state["newsletter_all_topics"]
        else st.session_state["newsletter_nb_topics"]
    )
    top_n_docs = (
        None
        if st.session_state["newsletter_all_docs"]
        else st.session_state["newsletter_nb_docs"]
    )

    return generate_newsletter(
        topic_model=st.session_state["topic_model"],
        df=df,
        topics=st.session_state["topics"],
        df_split=df_split,
        top_n_topics=top_n_topics,
        top_n_docs=top_n_docs,
        improve_topic_description=st.session_state["newsletter_improve_description"],
        summarizer_class=SUMMARIZER_OPTIONS_MAPPER[
            st.session_state["summarizer_classname"]
        ],
        summary_mode=st.session_state["summary_mode"],
    )


def main():
    # Check if a topic model exists
    if "topic_model" not in st.session_state:
        st.error(TRAIN_MODEL_FIRST_ERROR, icon=ERROR_ICON)
        st.stop()

    # Title
    st.title("Automatic newsletters generation")

    # Initialize session state variables
    default_values = {
        "newsletter_nb_topics": 4,
        "newsletter_nb_docs": 3,
        "summarizer_classname": list(SUMMARIZER_OPTIONS_MAPPER.keys())[0],
        "summary_mode": "topic",
        "newsletter_all_topics": False,
        "newsletter_all_docs": False,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Newsletter parameters sidebar
    with st.sidebar:
        register_multiple_widget(
            "newsletter_all_topics",
            "newsletter_all_docs",
            "newsletter_nb_topics",
            "newsletter_nb_docs",
            "newsletter_improve_description",
            "summarizer_classname",
            "summary_mode",
        )
        all_topics = st.checkbox(
            "Include all topics",
            on_change=save_widget_state,
            key="newsletter_all_topics",
        )
        st.slider(
            "Number of topics",
            min_value=1,
            max_value=20,
            on_change=save_widget_state,
            key="newsletter_nb_topics",
            disabled=all_topics,
        )

        all_documents = st.checkbox(
            "Include all documents per topic",
            on_change=save_widget_state,
            key="newsletter_all_docs",
        )
        st.slider(
            "Number of docs per topic",
            min_value=1,
            max_value=10,
            on_change=save_widget_state,
            key="newsletter_nb_docs",
            disabled=all_documents,
        )

        st.toggle(
            "Improve topic description",
            value=True,
            on_change=save_widget_state,
            key="newsletter_improve_description",
        )
        st.selectbox(
            "Summary mode",
            ["topic", "document", "none"],
            on_change=save_widget_state,
            key="summary_mode",
        )
        st.selectbox(
            "Summarizer class",
            list(SUMMARIZER_OPTIONS_MAPPER.keys()),
            on_change=save_widget_state,
            key="summarizer_classname",
        )

        generate_newsletter_clicked = st.button(
            "Generate newsletter", type="primary", use_container_width=True
        )

    # Generate newsletters when button is clicked
    if generate_newsletter_clicked:
        if st.session_state["split_type"] in ["yes", "enhanced"]:
            df = st.session_state["initial_df"]
            df_split = st.session_state["time_filtered_df"]
        else:
            df = st.session_state["time_filtered_df"]
            df_split = None

        with st.spinner("Generating newsletters..."):
            st.session_state["newsletters"] = generate_newsletter_wrapper(df, df_split)

    # Display generated newsletters
    if "newsletters" in st.session_state:
        st.components.v1.html(
            md2html(
                st.session_state["newsletters"][0],
                Path(inspect.getfile(generate_newsletter)).parent / "newsletter.css",
            ),
            height=800,
            scrolling=True,
        )


# Restore widget state
restore_widget_state()
main()
