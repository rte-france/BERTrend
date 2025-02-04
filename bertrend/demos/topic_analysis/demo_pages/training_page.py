#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import datetime

import pandas as pd
import streamlit as st
from loguru import logger

from bertrend import OUTPUT_PATH
from bertrend.demos.demos_utils.data_loading_component import (
    display_data_loading_component,
)
from bertrend.demos.demos_utils.embed_documents_component import (
    display_embed_documents_component,
)
from bertrend.demos.demos_utils.icons import (
    WARNING_ICON,
    SUCCESS_ICON,
    ERROR_ICON,
    SETTINGS_ICON,
    INFO_ICON,
    EMBEDDING_ICON,
)
from bertrend.demos.demos_utils.messages import (
    NO_EMBEDDINGS_WARNING_MESSAGE,
    MODEL_TRAINING_COMPLETE_MESSAGE,
)
from bertrend.demos.demos_utils.state_utils import (
    restore_widget_state,
    SessionStateManager,
)
from bertrend.demos.topic_analysis.messages import (
    EMBEDDINGS_CACHE_INFO,
    SAVE_MODEL_REMINDER,
    NO_MODEL_AVAILABLE_ERROR,
)
from bertrend.demos.demos_utils.parameters_component import (
    display_bertopic_hyperparameters,
    display_embedding_hyperparameters,
)
from bertrend.demos.weak_signals.visualizations_utils import PLOTLY_BUTTON_SAVE_CONFIG
from bertrend.metrics.topic_metrics import compute_cluster_metrics
from bertrend.config.parameters import BERTOPIC_SERIALIZATION
from bertrend.topic_analysis.visualizations import plot_docs_repartition_over_time
from bertrend.BERTopicModel import BERTopicModel
from bertrend.utils.data_loading import (
    TEXT_COLUMN,
)


def generate_model_name(base_name="topic_model"):
    """
    Generates a dynamic model name with the current date and time.
    If a base name is provided, it uses that instead of the default.
    """
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{base_name}_{current_datetime}"
    return model_name


def data_distribution(df: pd.DataFrame):
    """Display the distribution of data over time."""
    with st.expander(
        label="Data distribution",
        expanded=False,
    ):
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
        fig = plot_docs_repartition_over_time(df, freq)
        st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)


def save_model_interface():
    """Save the generated topic model to disk."""
    st.write("## Save Model")

    # Optional text box for custom model name
    base_model_name = st.text_input(
        "Enter a name for the model (optional):", key="base_model_name_input"
    )

    # Button to save the model
    if st.button("Save Model", key="save_model_button"):
        if "topic_model" in st.session_state:
            dynamic_model_name = generate_model_name(
                base_model_name if base_model_name else "topic_model"
            )
            model_save_path = OUTPUT_PATH / "saved_models" / dynamic_model_name
            logger.debug(
                f"Saving the model in the following directory: {model_save_path}"
            )
            try:
                st.session_state["topic_model"].save(
                    model_save_path,
                    serialization=BERTOPIC_SERIALIZATION,
                    save_ctfidf=True,
                    save_embedding_model=True,
                )
                st.success(
                    f"Model saved successfully as {model_save_path}", icon=SUCCESS_ICON
                )
                st.session_state["model_saved"] = True
                logger.success(f"Model saved successfully!", icon=SUCCESS_ICON)
            except Exception as e:
                st.error(f"Failed to save the model: {e}", icon=ERROR_ICON)
                logger.error(f"Failed to save the model: {e}")
        else:
            st.error(
                NO_MODEL_AVAILABLE_ERROR,
                icon=ERROR_ICON,
            )


def train_model():
    """Train a BERTopic model based on provided data."""
    with st.spinner("Training model..."):
        dataset = st.session_state["time_filtered_df"][TEXT_COLUMN]
        # indices = full_dataset.index.tolist()

        # Initialize topic model
        topic_model = BERTopicModel(st.session_state["bertopic_config"])
        embeddings = st.session_state["embeddings"]
        topic_model_output = topic_model.fit(
            docs=dataset,
            embedding_model=st.session_state["embedding_model_name"],
            embeddings=embeddings,
        )
        bertopic = topic_model_output.topic_model

        # Set session_state
        st.session_state["topic_model"] = bertopic
        st.session_state["topics"] = topic_model_output.topics

    st.success(MODEL_TRAINING_COMPLETE_MESSAGE, icon=SUCCESS_ICON)
    st.info(
        EMBEDDINGS_CACHE_INFO,
        icon=INFO_ICON,
    )

    topic_info = bertopic.get_topic_info()
    st.session_state["topics_info"] = topic_info[
        topic_info["Topic"] != -1
    ]  # exclude -1 topic from topic list

    # compute cluster metrics (optional)
    compute_cluster_metrics(bertopic, st.session_state["topics"], dataset)

    # update state
    st.session_state["model_trained"] = True
    if not st.session_state["model_saved"]:
        st.warning(SAVE_MODEL_REMINDER, icon=WARNING_ICON)


def main():
    st.title(":part_alternation_mark: Topic analysis demo")

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    if "model_saved" not in st.session_state:
        st.session_state["model_saved"] = False

    # In the sidebar form
    with st.sidebar:
        st.header(SETTINGS_ICON + " Settings")
        st.subheader(EMBEDDING_ICON + " Embedding Hyperparameters")
        display_embedding_hyperparameters()
        display_bertopic_hyperparameters()

    # Load data
    display_data_loading_component()

    # Data overview
    if "time_filtered_df" not in st.session_state:
        st.stop()
    data_distribution(st.session_state["time_filtered_df"])
    SessionStateManager.set("split_type", st.session_state["split_by_paragraph"])

    # Embed documents
    try:
        display_embed_documents_component()
    except Exception as e:
        logger.error(f"An error occurred while embedding documents: {str(e)}")
        st.error(
            f"An error occurred while embedding documents: {str(e)}", icon=ERROR_ICON
        )

    if not SessionStateManager.get("data_embedded", False):
        st.warning(NO_EMBEDDINGS_WARNING_MESSAGE, icon=WARNING_ICON)
        st.stop()

    # Train the model
    if st.button(
        "Train Model",
        type="primary",
        key="train_model_button",
        help="Make sure to review the settings before clicking on this button.",
    ):
        train_model()

        # Save the model
        save_model_interface()

        # TODO: Investigate the potentially deprecated save_model_interface() I implemented a while ago
        # to save a BERTopic model to either load it up later or load it up somewhere else


# Restore widget state
restore_widget_state()
main()
