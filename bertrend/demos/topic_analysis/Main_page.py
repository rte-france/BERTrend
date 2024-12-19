#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import ast
import datetime
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from bertrend import DATA_PATH

from bertrend.demos.topic_analysis.app_utils import (
    embedding_model_options,
    bertopic_options,
    umap_options,
    hdbscan_options,
    countvectorizer_options,
    ctfidf_options,
    representation_model_options,
    load_data_wrapper,
)
from bertrend.demos.topic_analysis.data_utils import data_overview, choose_data
from bertrend.demos.topic_analysis.state_utils import (
    register_widget,
    save_widget_state,
    restore_widget_state,
)

from bertrend.metrics.topic_metrics import get_coherence_value, get_diversity_value
from bertrend.train import train_BERTopic
from bertrend.utils.data_loading import (
    split_df_by_paragraphs,
    TIMESTAMP_COLUMN,
    URL_COLUMN,
    TEXT_COLUMN,
    preprocess_french_text,
    clean_dataset,
)


def split_dataframe(split_option, enhanced):
    """
    Split the dataframe based on the selected option.

    Args:
    split_option (str): The selected split option ('No split', 'Split by paragraphs')
    enhanced (bool): Whether to use enhanced splitting. Useful if we want to guarantee avoiding truncation
    during the embedding process, which happens if the input sequence length is more than the embedding model
    could handle.
    """
    if split_option == "No split":
        st.session_state["split_df"] = st.session_state["raw_df"]
        st.session_state["split_by_paragraphs"] = False
    else:  # Split by paragraph
        if enhanced:
            logger.debug(
                f"Using {st.session_state.get('embedding_model_name')} for enhanced splitting..."
            )
            model_name = st.session_state.get("embedding_model_name")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            max_length = SentenceTransformer(model_name).get_max_seq_length()

            # Correcting the max seq length anomaly in certain embedding models description
            if max_length == 514:
                max_length = 512

            with st.spinner("Splitting the dataset..."):
                st.session_state["split_df"] = split_df_by_paragraphs(
                    dataset=st.session_state["raw_df"],
                    enhanced=True,
                    tokenizer=tokenizer,
                    max_length=max_length
                    - 2,  # Minus 2 because beginning and end tokens are not considered
                    min_length=0,
                )
        else:
            st.session_state["split_df"] = split_df_by_paragraphs(
                st.session_state["raw_df"], enhanced=False
            )
        st.session_state["split_by_paragraphs"] = True


def generate_model_name(base_name="topic_model"):
    """
    Generates a dynamic model name with the current date and time.
    If a base name is provided, it uses that instead of the default.
    """
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{base_name}_{current_datetime}"
    return model_name


def save_model_interface():
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
            model_save_path = (
                Path(__file__).parent / "saved_models" / dynamic_model_name
            )
            logger.debug(
                f"Saving the model in the following directory: {model_save_path}"
            )
            try:
                st.session_state["topic_model"].save(
                    model_save_path,
                    serialization="safetensors",
                    save_ctfidf=True,
                    save_embedding_model=True,
                )
                st.success(f"Model saved successfully as {model_save_path}")
                st.session_state["model_saved"] = True
                logger.success(f"Model saved successfully!")
            except Exception as e:
                st.error(f"Failed to save the model: {e}")
                logger.error(f"Failed to save the model: {e}")
        else:
            st.error("No model available to save. Please train a model first.")


def select_data():
    st.write("## Data selection")

    choose_data(DATA_PATH, ["*.csv", "*.jsonl*", "*.parquet"])

    # Check if selected files have changed
    if (
        "previous_selected_files" not in st.session_state
        or st.session_state["previous_selected_files"]
        != st.session_state["selected_files"]
    ):
        st.session_state["previous_selected_files"] = st.session_state[
            "selected_files"
        ].copy()

        if st.session_state["selected_files"]:
            loaded_dfs = []
            for file_path in st.session_state["selected_files"]:
                df = load_data_wrapper(file_path)
                df.sort_values(by=TIMESTAMP_COLUMN, ascending=False, inplace=True)
                loaded_dfs.append(df)

            st.session_state["raw_df"] = (
                pd.concat(loaded_dfs) if len(loaded_dfs) > 1 else loaded_dfs[0]
            )

            # Add optional columns if not present
            if URL_COLUMN not in list(st.session_state["raw_df"].columns.values):
                logger.warning(f"{URL_COLUMN} not found in data")
                st.session_state["raw_df"][URL_COLUMN] = ""

            # Remove duplicates from raw_df
            st.session_state["raw_df"] = st.session_state["raw_df"].drop_duplicates(
                subset=TEXT_COLUMN, keep="first"
            )
            st.session_state["raw_df"].sort_values(
                by=[TIMESTAMP_COLUMN], ascending=True, inplace=True
            )
            st.session_state["initial_df"] = st.session_state["raw_df"].copy()

            # Reset the timestamp range when new files are selected
            min_max = st.session_state["raw_df"][TIMESTAMP_COLUMN].agg(["min", "max"])
            st.session_state["timestamp_range"] = (
                min_max["min"].to_pydatetime(),
                min_max["max"].to_pydatetime(),
            )

            # Trigger re-processing of the data
            st.session_state["data_changed"] = True
        else:
            st.error("Please select at least one file to proceed.")
            st.stop()

    st.divider()

    with st.container(border=True):
        # Select time range
        min_max = st.session_state["raw_df"][TIMESTAMP_COLUMN].agg(["min", "max"])
        register_widget("timestamp_range")
        timestamp_range = st.slider(
            "Select the range of timestamps you want to use for training",
            min_value=min_max["min"].to_pydatetime(),
            max_value=min_max["max"].to_pydatetime(),
            key="timestamp_range",
            on_change=save_widget_state,
        )

        # Filter text length parameter
        register_widget("min_text_length")
        min_text_length = st.number_input(
            "Select the minimum number of characters each document should contain",
            min_value=0,
            value=100,
            key="min_text_length",
            on_change=save_widget_state,
        )

        # Split or no split
        register_widget("split_option")
        split_options = ["No split", "Split by paragraphs"]
        split_option = st.radio(
            "Select the split option",
            split_options,
            key="split_option",
            on_change=save_widget_state,
            help="""
            - No split: No splitting on the documents.
            - Split by paragraphs: Split documents into paragraphs.
            """,
        )
        # Add a checkbox for enhanced splitting
        register_widget("enhanced_split")
        enhanced_split = st.checkbox(
            "Use enhanced splitting",
            key="enhanced_split",
            on_change=save_widget_state,
            help="If checked, uses a more advanced but slower method for splitting that considers the embedding model's maximum input length.",
        )

    # Check if any parameters have changed or if data has changed
    if (
        st.session_state.get("data_changed", False)
        or "split_method" not in st.session_state
        or st.session_state["split_method"] != split_option
        or "enhanced_splitting" not in st.session_state
        or st.session_state["enhanced_splitting"] != enhanced_split
        or "prev_timestamp_range" not in st.session_state
        or st.session_state["prev_timestamp_range"] != timestamp_range
        or "prev_min_text_length" not in st.session_state
        or st.session_state["prev_min_text_length"] != min_text_length
    ):
        st.session_state["split_method"] = split_option
        st.session_state["enhanced_splitting"] = enhanced_split
        st.session_state["prev_timestamp_range"] = timestamp_range
        st.session_state["prev_min_text_length"] = min_text_length

        if split_option != "No split":
            split_dataframe(split_option, enhanced_split)
        else:  # If No Splitting is done
            st.session_state["split_df"] = st.session_state["raw_df"]
            st.session_state["split_by_paragraphs"] = False

        # Preprocess the text
        st.session_state["split_df"][TEXT_COLUMN] = st.session_state["split_df"][
            TEXT_COLUMN
        ].apply(preprocess_french_text)

        # Remove unwanted rows from split_df
        st.session_state["split_df"] = st.session_state["split_df"][
            (st.session_state["split_df"][TEXT_COLUMN].str.strip() != "")
            & (
                st.session_state["split_df"][TEXT_COLUMN].apply(
                    lambda x: len(re.findall(r"[a-zA-Z]", x)) >= 5
                )
            )
        ]

        st.session_state["split_df"].reset_index(drop=True, inplace=True)
        st.session_state["split_df"]["index"] = st.session_state["split_df"].index

        # Filter dataset to select only text within time range
        st.session_state["timefiltered_df"] = st.session_state["split_df"].query(
            f"timestamp >= '{timestamp_range[0]}' and timestamp <= '{timestamp_range[1]}'"
        )

        # Clean dataset using min_text_length
        st.session_state["timefiltered_df"] = clean_dataset(
            st.session_state["timefiltered_df"],
            min_text_length,
        )

        st.session_state["timefiltered_df"] = (
            st.session_state["timefiltered_df"].reset_index(drop=True).reset_index()
        )

        # Reset the data_changed flag
        st.session_state["data_changed"] = False

    if st.session_state["timefiltered_df"].empty:
        st.error("Not enough remaining data after cleaning", icon="🚨")
        st.stop()
    else:
        st.info(
            f"Found {len(st.session_state['timefiltered_df'])} documents after final cleaning."
        )
        st.divider()


def train_model():
    if (
        "timefiltered_df" in st.session_state
        and not st.session_state["timefiltered_df"].empty
    ):
        with st.spinner("Training model..."):
            full_dataset = st.session_state["timefiltered_df"]
            indices = full_dataset.index.tolist()

            form_parameters = ast.literal_eval(st.session_state["parameters"])

            (
                st.session_state["topic_model"],
                st.session_state["topics"],
                _,
                st.session_state["embeddings"],
                st.session_state["token_embeddings"],
                st.session_state["token_strings"],
            ) = train_BERTopic(
                full_dataset=full_dataset,
                indices=indices,
                form_parameters=form_parameters,
                cache_base_name=(
                    st.session_state["data_name"]
                    if st.session_state["split_method"] == "No split"
                    else f'{st.session_state["data_name"]}_split_by_paragraphs'
                ),
            )

        st.success("Model trained successfully!")
        st.info(
            "Embeddings aren't saved in cache and thus aren't loaded. Please make sure to train the model without "
            "using cached embeddings if you want correct and functional temporal visualizations."
        )

        temp = st.session_state["topic_model"].get_topic_info()
        st.session_state["topics_info"] = temp[
            temp["Topic"] != -1
        ]  # exclude -1 topic from topic list

        # TOPIC MODEL COHERENCE AND DIVERSITY METRICS (optional) :
        coherence_score_type = "c_npmi"
        diversity_score_type = "puw"
        logger.info(
            f"Calculating {coherence_score_type} coherence and {diversity_score_type} diversity..."
        )

        try:
            coherence = get_coherence_value(
                st.session_state["topic_model"],
                st.session_state["topics"],
                st.session_state["timefiltered_df"][TEXT_COLUMN],
                coherence_score_type,
            )
            logger.success(f"Coherence score [{coherence_score_type}]: {coherence}")

        except IndexError as e:
            logger.error(
                "Error while calculating coherence metric. This likely happens when you're using an LLM to represent "
                "the topics instead of keywords."
            )
        try:
            diversity = get_diversity_value(
                st.session_state["topic_model"],
                st.session_state["topics"],
                st.session_state["timefiltered_df"][TEXT_COLUMN],
                diversity_score_type="puw",
            )
            logger.success(f"Diversity score [{diversity_score_type}]: {diversity}")
        except IndexError as e:
            logger.error(
                "Error while calculating diversity metric. This likely happens when you're using an LLM to represent "
                "the topics instead of keywords."
            )

        st.session_state["model_trained"] = True
        if not st.session_state["model_saved"]:
            st.warning("Don't forget to save your model!", icon="⚠️")
    else:
        st.error(
            "No data available for training. Please ensure data is correctly loaded."
        )


################################################
################## MAIN PAGE ###################
################################################

# Wide layout
st.set_page_config(page_title="BERTrend - Topic Analysis", layout="wide")

restore_widget_state()

### TITLE ###
st.title("Topic modelling")

if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False
if "model_saved" not in st.session_state:
    st.session_state["model_saved"] = False


def apply_changes():
    # Update other parameters
    parameters = {
        **embedding_model_options,
        **bertopic_options,
        **umap_options,
        **hdbscan_options,
        **countvectorizer_options,
        **ctfidf_options,
        **representation_model_options,
    }
    st.session_state["parameters"] = str(parameters)

    save_widget_state()
    st.sidebar.success("Changes applied successfully!")


# In the sidebar form
with st.sidebar.form("parameters_sidebar"):
    st.title("Parameters")

    with st.expander("Embedding model"):
        embedding_model_options = embedding_model_options()

    with st.expander("Topics"):
        bertopic_options = bertopic_options()

    with st.expander("UMAP"):
        umap_options = umap_options()

    with st.expander("HDBSCAN"):
        hdbscan_options = hdbscan_options()

    with st.expander("Count Vectorizer"):
        countvectorizer_options = countvectorizer_options()

    with st.expander("c-TF-IDF"):
        ctfidf_options = ctfidf_options()

    with st.expander("Representation Models"):
        representation_model_options = representation_model_options()

        # Form submit button for applying changes
    # (using on_click with callback function causes a glitch where the button has to be clicked twice for changes to take effect)
    changes_applied = st.form_submit_button(
        label="Apply Changes", type="primary", use_container_width=True
    )
    if changes_applied:
        apply_changes()

# Separate button for training the model
if st.sidebar.button(
    "Train Model",
    type="primary",
    key="train_model_button",
    use_container_width=True,
    disabled=("parameters" not in st.session_state),
    help="Make sure to review and apply changes before clicking on this button.",
):
    train_model()

if "parameters" in st.session_state:
    st.sidebar.write(f"Current parameters:")
    st.sidebar.write(st.session_state["parameters"])

# Load selected DataFrame
select_data()

# Data overview
data_overview(st.session_state["timefiltered_df"])

# Save the model button
save_model_interface()

# TODO: Investigate the potentially deprecated save_model_interface() I implemented a while ago
# to save a BERTopic model to either load it up later or load it up somewhere else
