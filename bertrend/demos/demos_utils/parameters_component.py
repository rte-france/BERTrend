#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st

from code_editor import code_editor

from bertrend import EMBEDDING_CONFIG
from bertrend.demos.demos_utils.state_utils import (
    register_widget,
    save_widget_state,
    SessionStateManager,
    register_multiple_widget,
    reset_widget_state,
)
from bertrend.parameters import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_ZEROSHOT_MIN_SIMILARITY,
    EMBEDDING_DTYPES,
    LANGUAGES,
    ENGLISH_EMBEDDING_MODELS,
    FRENCH_EMBEDDING_MODELS,
    REPRESENTATION_MODELS,
    MMR_REPRESENTATION_MODEL,
    DEFAULT_BERTOPIC_CONFIG_FILE,
)

from bertrend.demos.demos_utils.icons import INFO_ICON


def display_local_embeddings():
    """UI settings for local embedding service"""
    register_multiple_widget("language", "embedding_dtype", "embedding_model_name")

    language = st.selectbox(
        "Select Language",
        LANGUAGES,
        key="language",
        on_change=_on_language_change,
    )
    st.selectbox(
        "Embedding Dtype",
        EMBEDDING_DTYPES,
        key="embedding_dtype",
        on_change=save_widget_state,
    )
    embedding_models = (
        ENGLISH_EMBEDDING_MODELS if language == "English" else FRENCH_EMBEDDING_MODELS
    )
    st.selectbox(
        "Embedding Model",
        options=embedding_models,
        key="embedding_model_name",
        on_change=save_widget_state,
    )


def _on_language_change():
    save_widget_state()
    # required to handle multi-page app and restore_state call:
    reset_widget_state("embedding_model_name")
    st.session_state.pop("embedding_model_name")


def display_remote_embeddings():
    """UI settings for remote embedding service"""
    register_widget("embedding_service_hostname")
    register_widget("embedding_service_port")
    if "embedding_service_hostname" not in st.session_state:
        st.session_state["embedding_service_hostname"] = EMBEDDING_CONFIG["host"]
    if "embedding_service_port" not in st.session_state:
        st.session_state["embedding_service_port"] = EMBEDDING_CONFIG["port"]
    st.text_input(
        "Embedding service hostname",
        key="embedding_service_hostname",
        on_change=save_widget_state,
    )
    st.text_input(
        "Embedding service port",
        value=EMBEDDING_CONFIG["port"],
        on_change=save_widget_state,
    )


def display_bertopic_hyperparameters():
    # Embedding model parameters
    with st.expander("Embedding Model Settings", expanded=False):
        register_widget("embedding_service_type")
        if "embedding_service_type" not in st.session_state:
            st.session_state["embedding_service_type"] = "local"
        st.segmented_control(
            "Embedding service",
            selection_mode="single",
            key="embedding_service_type",
            options=["local", "remote"],
            on_change=save_widget_state,
        )
        if st.session_state["embedding_service_type"] == "local":
            display_local_embeddings()
        else:
            display_remote_embeddings()

    # BERTopic model parameters
    with st.expander("BERTopic Model Settings", expanded=False):
        # Get BERTopic default configuration
        with open(DEFAULT_BERTOPIC_CONFIG_FILE, "r") as f:
            # Load default parameter the first time
            toml_txt = f.read()

        # Add code editor to edit the config file
        st.write(INFO_ICON + " CTRL + Enter to update")
        config_editor = code_editor(toml_txt, lang="yaml")

        # If code is edited, update config
        if config_editor["text"] != "":
            st.session_state["bertopic_config"] = config_editor["text"]
        # Else use default config
        else:
            st.session_state["bertopic_config"] = toml_txt


def display_bertrend_hyperparameters():
    """UI settings for Bertrend hyperparameters"""
    with st.expander("Merging Hyperparameters", expanded=False):
        register_widget("min_similarity")
        st.slider(
            "Minimum Similarity for Merging",
            0.0,
            1.0,
            DEFAULT_MIN_SIMILARITY,
            0.01,
            key="min_similarity",
            on_change=save_widget_state,
        )

    with st.expander("Zero-shot Parameters", expanded=False):
        register_widget("zeroshot_min_similarity")
        st.slider(
            "Zeroshot Minimum Similarity",
            0.0,
            1.0,
            DEFAULT_ZEROSHOT_MIN_SIMILARITY,
            0.01,
            key="zeroshot_min_similarity",
            on_change=save_widget_state,
        )


def display_representation_model_options():
    """UI settings for representation model options"""
    with st.expander("Representation model selection", expanded=False):
        register_widget("representation_models")
        selected_models = st.multiselect(
            label="Select representation models",
            options=REPRESENTATION_MODELS,
            default=MMR_REPRESENTATION_MODEL,
            key="representation_models",
            on_change=save_widget_state,
        )
