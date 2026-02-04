#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import torch

# workaround with streamlit to avoid errors Examining the path of torch.classes raised: Tried to instantiate class 'path.path’, but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

import pickle
import shutil
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from bertrend import CACHE_PATH, ZEROSHOT_TOPICS_DATA_DIR
from bertrend.BERTopicModel import BERTopicModel
from bertrend.BERTrend import BERTrend
from bertrend.config.parameters import *
from bertrend.demos.demos_utils import is_admin_mode
from bertrend.demos.demos_utils.data_loading_component import (
    display_data_loading_component,
)
from bertrend.demos.demos_utils.embed_documents_component import (
    display_embed_documents_component,
)
from bertrend.demos.demos_utils.i18n import (
    create_internationalization_language_selector,
    translate,
)
from bertrend.demos.demos_utils.icons import (
    ANALYSIS_ICON,
    DATA_LOADING_ICON,
    EMBEDDING_ICON,
    ERROR_ICON,
    MODEL_TRAINING_ICON,
    SETTINGS_ICON,
    SUCCESS_ICON,
    TOPIC_ICON,
    TREND_ICON,
    WARNING_ICON,
)
from bertrend.demos.demos_utils.parameters_component import (
    display_bertopic_hyperparameters,
    display_bertrend_hyperparameters,
    display_embedding_hyperparameters,
)
from bertrend.demos.demos_utils.state_utils import SessionStateManager
from bertrend.demos.weak_signals.visualizations_utils import (
    PLOTLY_BUTTON_SAVE_CONFIG,
    display_newly_emerged_topics,
    display_popularity_evolution,
    display_sankey_diagram,
    display_signal_analysis,
    display_signal_types,
    display_topics_per_timestamp,
    retrieve_topic_counts,
    save_signal_evolution,
)
from bertrend.trend_analysis.visualizations import (
    plot_num_topics,
    plot_size_outliers,
)
from bertrend.trend_analysis.weak_signals import detect_weak_signals_zeroshot
from bertrend.utils.data_loading import (
    TEXT_COLUMN,
    group_by_days,
)


# UI Settings
def PAGE_TITLE():
    return translate("page_title")


LAYOUT: Literal["centered", "wide"] = "wide"


# TODO: handle uploaded files
def save_state():
    """Save the application state"""
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    # Save the selected files (list of filenames)
    selected_files = SessionStateManager.get("selected_files", [])

    state = SessionStateManager.get_multiple(
        "selected_files",
        "min_chars",
        "split_by_paragraph",
        "timeframe_slider",
        "language",
        "embedding_model_name",
        "embedding_model",
        "sample_size",
        "min_similarity",
        "zeroshot_min_similarity",
        "embedding_dtype",
        "data_embedded",
    )

    state["selected_files"] = selected_files

    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    np.save(embeddings_file, SessionStateManager.get_embeddings())
    st.success(translate("state_saved_message"), icon=SUCCESS_ICON)


# TODO: handle uploaded files
def restore_state():
    """Restore the application state"""
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    if state_file.exists() and embeddings_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)

        # Restore the selected files
        selected_files = state.get("selected_files", [])
        SessionStateManager.set("selected_files", selected_files)

        # Restore other states
        SessionStateManager.set_multiple(**state)
        SessionStateManager.set("embeddings", np.load(embeddings_file))
        st.success(translate("state_restored_message"), icon=SUCCESS_ICON)

        # Update the multiselect widget with restored selected files
        st.session_state["selected_files"] = selected_files
    else:
        st.warning(translate("no_state_warning"), icon=WARNING_ICON)


def purge_cache():
    """Purge cache data"""
    if CACHE_PATH.exists():
        shutil.rmtree(CACHE_PATH)
        st.success(translate("cache_purged_message"), icon=SUCCESS_ICON)
    else:
        st.warning(translate("no_cache_warning"), icon=WARNING_ICON)


def load_data_page():
    st.header(translate("data_loading_and_preprocessing"))

    display_data_loading_component()

    if "time_filtered_df" in st.session_state:
        try:
            display_embed_documents_component()
            if SessionStateManager.get("data_embedded", False):
                save_state()
        except Exception as e:
            logger.error(f"An error occurred while embedding documents: {e}")
            st.error(
                translate("error_embedding_documents").format(e=e),
                icon=ERROR_ICON,
            )


def training_page():
    st.header(translate("model_training"))

    if not SessionStateManager.get("data_embedded"):
        st.warning(translate("no_embeddings_warning_message"), icon=WARNING_ICON)
        st.stop()

    # Show documents per grouped timestamp
    with st.expander(translate("documents_per_timestamp"), expanded=True):
        st.write(f"{translate('granularity')}: {st.session_state['granularity']}")
        grouped_data = group_by_days(
            SessionStateManager.get_dataframe("time_filtered_df"),
            day_granularity=st.session_state["granularity"],
        )
        non_empty_timestamps = [
            timestamp for timestamp, group in grouped_data.items() if not group.empty
        ]
        if non_empty_timestamps:
            selected_timestamp = st.select_slider(
                translate("select_timestamp"),
                options=non_empty_timestamps,
                key="timestamp_slider",
            )
            selected_docs = grouped_data[selected_timestamp]
            st.dataframe(
                selected_docs[
                    ["timestamp", TEXT_COLUMN, "document_id", "source", "url"]
                ],
                width="stretch",
            )
        else:
            st.warning(translate("no_data_warning"), icon=WARNING_ICON)

    if not SessionStateManager.get("data_embedded", False):
        st.warning(
            translate("embed_warning"),
            icon=WARNING_ICON,
        )
        st.stop()
    else:
        # Zero-shot topic definition
        zeroshot_topic_list = st.text_input(
            translate("enter_zeroshot_topics"), value=""
        )
        zeroshot_topic_list = [
            topic.strip() for topic in zeroshot_topic_list.split("/") if topic.strip()
        ]
        SessionStateManager.set("zeroshot_topic_list", zeroshot_topic_list)

        if st.button(translate("train_models"), type="primary"):
            with st.spinner(translate("training_models")):
                # FIXME: called twice (see above)
                grouped_data = group_by_days(
                    SessionStateManager.get_dataframe("time_filtered_df"),
                    day_granularity=st.session_state["granularity"],
                )

                # Initialize topic model
                topic_model = BERTopicModel(st.session_state["bertopic_config"])

                # Created BERTrend object
                bertrend = BERTrend(
                    config_file=st.session_state["bertrend_config"],
                    topic_model=topic_model,
                )
                # Train topic models on data
                bertrend.train_topic_models(
                    grouped_data=grouped_data,
                    embedding_model=SessionStateManager.get("embedding_model"),
                    embeddings=SessionStateManager.get_embeddings(),
                )
                st.success(
                    translate("model_training_complete_message"), icon=SUCCESS_ICON
                )

                # Save trained models
                bertrend.save_model()
                st.success(translate("models_saved_message"), icon=SUCCESS_ICON)

                # Compute signal popularity
                bertrend.calculate_signal_popularity()
                SessionStateManager.set("popularity_computed", True)

                # Store bertrend object
                SessionStateManager.set("bertrend", bertrend)

                st.success(
                    translate("model_merging_complete_message"), icon=SUCCESS_ICON
                )


def analysis_page():
    st.header(translate("results_analysis"))

    if not SessionStateManager.get("data_embedded"):
        st.warning(
            translate("embed_train_warning"),
            icon=WARNING_ICON,
        )
        st.stop()

    elif (
        not SessionStateManager.get("bertrend")
        or not SessionStateManager.get("bertrend")._is_fitted
    ):
        st.warning(
            translate("train_warning"),
            icon=WARNING_ICON,
        )
        st.stop()

    else:
        topic_models = SessionStateManager.get("bertrend").restore_topic_models()
        with st.expander(translate("topic_overview"), expanded=False):
            # Number of Topics Detected for each topic model
            st.plotly_chart(
                plot_num_topics(topic_models),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                width="stretch",
            )
            # Size of Outlier Topic for each topic model
            st.plotly_chart(
                plot_size_outliers(topic_models),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                width="stretch",
            )

        display_topics_per_timestamp(topic_models)

        # Display zeroshot signal trend
        zeroshot_topic_list = SessionStateManager.get("zeroshot_topic_list", None)
        if zeroshot_topic_list:
            st.subheader(translate("zeroshot_weak_signal_trends"))
            weak_signal_trends = detect_weak_signals_zeroshot(
                topic_models,
                zeroshot_topic_list,
                st.session_state["granularity"],
            )
            with st.expander(translate("zeroshot_weak_signal_trends"), expanded=False):
                fig_trend = go.Figure()
                for topic, weak_signal_trend in weak_signal_trends.items():
                    timestamps = list(weak_signal_trend.keys())
                    popularity = [
                        weak_signal_trend[timestamp]["Document_Count"]
                        for timestamp in timestamps
                    ]
                    hovertext = [
                        f"Topic: {topic}<br>{translate('timestamp')}: {timestamp}<br>{translate('popularity')}: {weak_signal_trend[timestamp]['Document_Count']}<br>Representation: {weak_signal_trend[timestamp]['Representation']}"
                        for timestamp in timestamps
                    ]
                    fig_trend.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=popularity,
                            mode="lines+markers",
                            name=topic,
                            hovertext=hovertext,
                            hoverinfo="text",
                        )
                    )
                fig_trend.update_layout(
                    title=translate("popularity_of_zeroshot_topics"),
                    xaxis_title=translate("timestamp"),
                    yaxis_title=translate("popularity"),
                )
                st.plotly_chart(
                    fig_trend,
                    config=PLOTLY_BUTTON_SAVE_CONFIG,
                    width="stretch",
                )

                # Display the dataframe with zeroshot topics information
                zeroshot_topics_data = [
                    {
                        "Topic": topic,
                        "Timestamp": timestamp,
                        "Representation": data["Representation"],
                        "Representative_Docs": data["Representative_Docs"],
                        "Count": data["Count"],
                        "Document_Count": data["Document_Count"],
                    }
                    for topic, weak_signal_trend in weak_signal_trends.items()
                    for timestamp, data in weak_signal_trend.items()
                ]
                zeroshot_topics_df = pd.DataFrame(zeroshot_topics_data)
                st.dataframe(zeroshot_topics_df, width="stretch")

                # Save the zeroshot topics data to a JSON file
                json_file_path = ZEROSHOT_TOPICS_DATA_DIR
                json_file_path.mkdir(parents=True, exist_ok=True)

                zeroshot_topics_df.to_json(
                    json_file_path / ZEROSHOT_TOPICS_DATA_FILE,
                    orient="records",
                    date_format="iso",
                    indent=4,
                )
                st.success(
                    translate("zeroshot_topics_data_saved").format(
                        json_file_path=json_file_path
                    ),
                    icon=SUCCESS_ICON,
                )

        if not SessionStateManager.get("popularity_computed", False):
            st.warning(
                translate("merge_warning"),
                icon=WARNING_ICON,
            )
            st.stop()

        else:
            # Display merged signal trend
            with st.expander(translate("topic_size_evolution"), expanded=False):
                st.dataframe(
                    SessionStateManager.get("bertrend").all_merge_histories_df[
                        [
                            "Timestamp",
                            "Topic1",
                            "Topic2",
                            "Representation1",
                            "Representation2",
                            "Document_Count1",
                            "Document_Count2",
                        ]
                    ]
                )

            # Display topic popularity evolution
            with st.expander(translate("topic_popularity_evolution"), expanded=True):
                display_popularity_evolution()
                # Save Signal Evolution Data to investigate later on in a separate notebook
                save_signal_evolution()

            # Show weak/strong signals
            display_signal_types()

            # Analyze signal
            with st.expander(translate("signal_analysis"), expanded=True):
                st.subheader(translate("signal_analysis"))
                topic_number = st.number_input(
                    translate("enter_topic_number"), min_value=0, step=1
                )
                if st.button(translate("analyze_signal"), type="primary"):
                    try:
                        display_signal_analysis(topic_number)
                    except Exception as e:
                        st.error(
                            translate("error_generating_signal_summary").format(e=e),
                            icon=ERROR_ICON,
                        )

            # Create the Sankey Diagram
            st.subheader(translate("topic_evolution"))
            display_sankey_diagram(
                SessionStateManager.get("bertrend").all_merge_histories_df
            )

            # Newly emerged topics
            if SessionStateManager.get("bertrend").all_new_topics_df is not None:
                st.subheader(translate("newly_emerged_topics"))
                display_newly_emerged_topics(
                    SessionStateManager.get("bertrend").all_new_topics_df
                )

            if st.button(translate("retrieve_topic_counts")):
                with st.spinner(translate("retrieving_topic_counts")):
                    # Number of topics per individual topic model
                    retrieve_topic_counts(topic_models)


def main():
    st.set_page_config(
        page_title=PAGE_TITLE(),
        layout=LAYOUT,
        initial_sidebar_state="expanded" if is_admin_mode() else "collapsed",
        page_icon=":part_alternation_mark:",
    )

    st.title(":part_alternation_mark: " + PAGE_TITLE())

    # Set the main flags
    SessionStateManager.get_or_set("data_embedded", False)
    SessionStateManager.get_or_set("popularity_computed", False)

    # Sidebar
    with st.sidebar:
        # Add language selector
        create_internationalization_language_selector()

        st.header(SETTINGS_ICON + " " + translate("settings_and_controls"))

        # State Management
        st.subheader(translate("state_management"))

        if st.button(translate("restore_previous_run"), width="stretch"):
            restore_state()
            try:
                SessionStateManager.set("bertrend", BERTrend.restore_model())
                st.success(translate("models_restored_message"), icon=SUCCESS_ICON)
            except Exception:
                st.warning(translate("no_models_warning"), icon=WARNING_ICON)

        if st.button(translate("purge_cache"), width="stretch"):
            purge_cache()

        if st.button(translate("clear_session_state"), width="stretch"):
            SessionStateManager.clear()

        # BERTopic Hyperparameters
        st.subheader(EMBEDDING_ICON + " " + translate("embedding_hyperparameters"))
        display_embedding_hyperparameters()
        st.subheader(TOPIC_ICON + " " + translate("bertopic_hyperparameters"))
        display_bertopic_hyperparameters()
        st.subheader(TREND_ICON + " " + translate("bertrend_hyperparameters"))
        display_bertrend_hyperparameters()

    # Main content
    tab1, tab2, tab3 = st.tabs(
        [
            DATA_LOADING_ICON + " " + translate("data_loading"),
            MODEL_TRAINING_ICON + " " + translate("model_training"),
            ANALYSIS_ICON + " " + translate("results_analysis"),
        ]
    )

    with tab1:
        load_data_page()

    with tab2:
        training_page()

    with tab3:
        analysis_page()


if __name__ == "__main__":
    main()
