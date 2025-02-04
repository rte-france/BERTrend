#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st

from code_editor import code_editor

from bertrend import (
    BERTOPIC_DEFAULT_CONFIG_PATH,
    BERTREND_DEFAULT_CONFIG_PATH,
    EMBEDDING_CONFIG,
    load_toml_config,
)
from bertrend.demos.demos_utils.state_utils import (
    register_widget,
    save_widget_state,
    SessionStateManager,
    register_multiple_widget,
    reset_widget_state,
)
from bertrend.config.parameters import (
    EMBEDDING_DTYPES,
    LANGUAGES,
    ENGLISH_EMBEDDING_MODELS,
    FRENCH_EMBEDDING_MODELS,
    REPRESENTATION_MODELS,
    MMR_REPRESENTATION_MODEL,
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


def display_embedding_hyperparameters():
    """UI settings for embedding hyperparameters"""
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


def display_bertopic_hyperparameters():
    # BERTopic model parameters
    with st.expander("BERTopic Model Settings", expanded=False):
        # If BERTopic config is already in session state, use it
        if "bertopic_config" in st.session_state:
            toml_txt = st.session_state["bertopic_config"]
        # Else get BERTopic default configuration
        else:
            with open(BERTOPIC_DEFAULT_CONFIG_PATH, "r") as f:
                # Load default parameter the first time
                toml_txt = f.read()

        # Add code editor to edit the config file
        st.write(INFO_ICON + " CTRL + Enter to update")
        config_editor = code_editor(
            toml_txt,
            lang="toml",
        )

        # If code is edited, update config
        if config_editor["text"] != "":
            st.session_state["bertopic_config"] = config_editor["text"]
        # Else use default config
        else:
            st.session_state["bertopic_config"] = toml_txt


def display_bertrend_hyperparameters():
    """UI settings for Bertrend hyperparameters"""
    with st.expander("BERTrend Model Settings", expanded=False):
        # Get BERTrend default configuration
        with open(BERTREND_DEFAULT_CONFIG_PATH, "r") as f:
            # Load default parameter the first time
            toml_txt = f.read()

        # Add code editor to edit the config file
        st.write(INFO_ICON + " CTRL + Enter to update")
        config_editor = code_editor(toml_txt, lang="toml")

        # If code is edited, update config
        if config_editor["text"] != "":
            st.session_state["bertrend_config"] = config_editor["text"]
        # Else use default config
        else:
            st.session_state["bertrend_config"] = toml_txt

        # Save granularity in session state as it is re-used in other components
        st.session_state["granularity"] = load_toml_config(
            st.session_state["bertrend_config"]
        )["granularity"]


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
