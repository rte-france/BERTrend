#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import streamlit as st
from pathlib import Path
from bertopic import BERTopic
from typing import List, Optional, Union

from bertrend.demos.topic_analysis.state_utils import restore_widget_state


def list_saved_models(saved_models_dir: Union[str, Path]) -> List[Path]:
    """
    Lists all saved models in the given directory.

    Args:
    saved_models_dir (Union[str, Path]): The directory where models are saved.

    Returns:
    List[Path]: A list of paths to the saved models.
    """
    saved_models_dir = Path(saved_models_dir)
    if not saved_models_dir.exists():
        st.error(f"The directory {saved_models_dir} does not exist.")
        return []

    return [model for model in saved_models_dir.glob("*") if model.is_dir()]


def load_and_merge_models(
    selected_models_order: List[str], min_similarity: float
) -> Optional[BERTopic]:
    loaded_models: List[BERTopic] = []
    failed_to_load: List[str] = []

    for i, model_name in enumerate(selected_models_order):
        try:
            model_path = f"./saved_models/{model_name}"
            loaded_model = BERTopic.load(model_path)
            loaded_models.append(loaded_model)
            st.session_state[f"loaded_model_{i+1}"] = loaded_model
        except Exception as e:
            failed_to_load.append(model_name)

    if failed_to_load:
        st.error(f"Failed to load the following model(s): {', '.join(failed_to_load)}")
        return None

    st.success("All models were loaded successfully.")

    try:
        merged_model = BERTopic.merge_models(
            loaded_models, min_similarity=min_similarity
        )
        st.success("Models were merged successfully.")
        return merged_model
    except Exception as e:
        st.error("Failed to merge models.")
        return None


def display_model_checkboxes(models: List[Path]) -> None:
    """
    Displays checkboxes for each model, updates selection order immediately,
    requires at least two models to be selected for merging, attempts to load the selected models,
    displays success/error messages, and merges them if all are loaded successfully.
    """
    if not models:
        st.info("No models found.")
        return []

    model_names = [model.name for model in models]

    if "selected_models_order" not in st.session_state:
        st.session_state["selected_models_order"] = []

    for model_name in model_names:
        if st.checkbox(model_name, key=model_name):
            if model_name not in st.session_state["selected_models_order"]:
                st.session_state["selected_models_order"].append(model_name)
        else:
            if model_name in st.session_state["selected_models_order"]:
                st.session_state["selected_models_order"].remove(model_name)

    selected_models_order = st.session_state["selected_models_order"]

    if selected_models_order:
        st.markdown("### Selection Order:")
        for i, model_name in enumerate(selected_models_order, start=1):
            st.markdown(f"**{i}. {model_name}**")

    if len(st.session_state.get("selected_models_order", [])) >= 2:
        if st.button("Merge"):
            merged_model = load_and_merge_models(
                st.session_state["selected_models_order"],
                st.session_state["min_similarity"],
            )
            if merged_model:
                st.session_state["merged_model"] = merged_model
                st.session_state["merge_successful"] = True
                # Reset the warning flag here to remove the warning after a successful merge
                st.session_state["min_similarity_changed"] = False
    else:
        st.info("Please select at least two models to enable merging.")


def display_merged_model_overview() -> None:
    if "merge_successful" in st.session_state and st.session_state["merge_successful"]:
        with st.expander("Overview", expanded=True):
            merged_model = st.session_state.get("merged_model", None)
            first_loaded_model = st.session_state.get("loaded_model_1", None)

            if merged_model and first_loaded_model:
                st.markdown("#### Number of Topics")
                num_topics_original = len(first_loaded_model.get_topic_info())
                num_topics_merged = len(merged_model.get_topic_info())
                num_new_topics = num_topics_merged - num_topics_original

                # Displaying overview statistics with bullet points and bold font for labels
                st.markdown(
                    f"- **Number of topics in original model:** {num_topics_original}"
                )
                st.markdown(
                    f"- **Number of topics in merged model:** {num_topics_merged}"
                )
                st.markdown(
                    f"- **Number of new topics in merged model:** {num_new_topics}"
                )

                # Displaying new topics if there are any
                if num_new_topics > 0:
                    new_topics_df = merged_model.get_topic_info().tail(num_new_topics)
                    st.markdown("#### New Topics in Merged Model")
                    st.table(new_topics_df)
            else:
                # This condition is only visible if there's an attempt to access
                # the merged model overview and there's an actual issue with model access.
                if "merge_successful" in st.session_state and not st.session_state.get(
                    "merged_model", None
                ):
                    st.error(
                        "Error accessing merged model or original model information."
                    )


def on_min_similarity_change():
    # Check if a merge has been done before showing the warning
    if st.session_state.get("merge_successful", False):
        st.session_state["min_similarity_changed"] = True


#########################################
########## MERGE MODEL PAGE #############
#########################################

# Page Configuration
st.set_page_config(page_title="Merge BERTopic Models", layout="wide")

# Restore widget state
restore_widget_state()

# Initialize session state variables on first run
if "app_started" not in st.session_state:
    st.session_state.app_started = True
    st.session_state.merge_successful = False
    st.session_state.min_similarity_changed = False
    st.session_state.selected_models_order = []


# Min_similarity slider with callback
min_similarity = st.sidebar.slider(
    "min_similarity",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    key="min_similarity",
    on_change=on_min_similarity_change,
)


# Display model selection checkboxes
display_model_checkboxes(models=list_saved_models("./saved_models"))

# Display merged model overview in an expander
display_merged_model_overview()

# Reset flags upon rerun to clear overview and remove warning
st.sidebar.button(
    "Clear Overview & Reset",
    on_click=lambda: [
        setattr(st.session_state, key, False)
        for key in ["merge_successful", "min_similarity_changed"]
    ],
)

# Warning for min_similarity change after merge
if st.session_state.get("min_similarity_changed", False):
    st.warning(
        "You've changed the 'min_similarity' value. Please perform the merge again to see the updated results.",
        icon="⚠️",
    )
