import datetime

import pandas as pd
import streamlit as st
from loguru import logger

from bertrend import OUTPUT_PATH
from bertrend.BERTopicModel import BERTopicModel
from bertrend.config.parameters import BERTOPIC_SERIALIZATION
from bertrend.demos.demos_utils.data_loading_component import (
    display_data_loading_component,
)
from bertrend.demos.demos_utils.embed_documents_component import (
    display_embed_documents_component,
)
from bertrend.demos.demos_utils.i18n import translate
from bertrend.demos.demos_utils.icons import (
    EMBEDDING_ICON,
    ERROR_ICON,
    INFO_ICON,
    SETTINGS_ICON,
    SUCCESS_ICON,
    TOPIC_ICON,
    WARNING_ICON,
)
from bertrend.demos.demos_utils.parameters_component import (
    display_bertopic_hyperparameters,
    display_embedding_hyperparameters,
)
from bertrend.demos.demos_utils.state_utils import (
    SessionStateManager,
    restore_widget_state,
)
from bertrend.demos.weak_signals.visualizations_utils import PLOTLY_BUTTON_SAVE_CONFIG
from bertrend.metrics.topic_metrics import compute_cluster_metrics
from bertrend.topic_analysis.visualizations import plot_docs_repartition_over_time
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
        label=translate("data_distribution"),
        expanded=False,
    ):
        freq = st.select_slider(
            translate("time_aggregation"),
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
        st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, width="stretch")


def save_model_interface():
    """Save the generated topic model to disk."""
    st.write("## " + translate("save_model"))

    # Optional text box for custom model name
    base_model_name = st.text_input(
        translate("enter_model_name"), key="base_model_name_input"
    )

    # Button to save the model
    if st.button(translate("save_model"), key="save_model_button"):
        if "topic_model" in st.session_state:
            dynamic_model_name = generate_model_name(base_model_name or "topic_model")
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
                    translate("model_saved_successfully").format(
                        model_save_path=model_save_path
                    ),
                    icon=SUCCESS_ICON,
                )
                st.session_state["model_saved"] = True
                logger.success("Model saved successfully!", icon=SUCCESS_ICON)
            except Exception as e:
                st.error(f"Failed to save the model: {e}", icon=ERROR_ICON)
                logger.error(f"Failed to save the model: {e}")
        else:
            st.error(
                translate("no_model_available_error"),
                icon=ERROR_ICON,
            )


def train_model():
    """Train a BERTopic model based on provided data."""
    with st.spinner(translate("training_model")):
        dataset = st.session_state["time_filtered_df"][TEXT_COLUMN]
        # indices = full_dataset.index.tolist()

        # Initialize topic model
        topic_model = BERTopicModel(st.session_state["bertopic_config"])
        embeddings = st.session_state["embeddings"]
        topic_model_output = topic_model.fit(
            docs=dataset,
            embeddings=embeddings,
            embedding_model=SessionStateManager.get("embedding_model_name"),
        )
        bertopic = topic_model_output.topic_model

        # Set session_state
        st.session_state["topic_model"] = bertopic
        st.session_state["topics"] = topic_model_output.topics

    st.success(translate("model_training_complete_message"), icon=SUCCESS_ICON)
    st.info(
        translate("embeddings_cache_info"),
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
        st.warning(translate("save_model_reminder"), icon=WARNING_ICON)


def main():
    st.title(translate("topic_analysis_demo_title"))

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    if "model_saved" not in st.session_state:
        st.session_state["model_saved"] = False

    # In the sidebar form
    with st.sidebar:
        st.header(SETTINGS_ICON + " " + translate("settings_and_controls"))
        st.subheader(EMBEDDING_ICON + " " + translate("embedding_hyperparameters"))
        display_embedding_hyperparameters()
        st.subheader(TOPIC_ICON + " " + translate("bertopic_hyperparameters"))
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
        logger.error(f"An error occurred while embedding documents: {e}")
        st.error(translate("error_embedding_documents").format(e=e), icon=ERROR_ICON)

    if not SessionStateManager.get("data_embedded", False):
        st.warning(translate("no_embeddings_warning_message"), icon=WARNING_ICON)
        st.stop()

    # Train the model
    if st.button(
        translate("train_model"),
        type="primary",
        key="train_model_button",
        help=translate("review_settings_help"),
    ):
        train_model()

        # Save the model
        save_model_interface()

        # TODO: Investigate the potentially deprecated save_model_interface() I implemented a while ago
        # to save a BERTopic model to either load it up later or load it up somewhere else


# Restore widget state
restore_widget_state()
main()
