#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from bertrend.demos.topic_analysis.state_utils import register_widget
from bertrend.demos.weak_signals.visualizations_utils import PLOTLY_BUTTON_SAVE_CONFIG
from bertrend.utils.data_loading import (
    load_data,
    TIMESTAMP_COLUMN,
    GROUPED_TIMESTAMP_COLUMN,
    TEXT_COLUMN,
    TITLE_COLUMN,
    URL_COLUMN,
    CITATION_COUNT_COL,
)

# Default configuration parameters for the application
DEFAULT_PARAMETERS = {
    "embedding_model_name": "OrdalieTech/Solon-embeddings-base-0.1",
    "use_cached_embeddings": False,
    "bertopic_nr_topics": 0,
    "bertopic_top_n_words": 10,
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 10,
    "hdbscan_metric": "euclidean",
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_cluster_selection_epsilon": 0.0,
    "hdbscan_max_cluster_size": 0,
    "hdbscan_allow_single_cluster": False,
    "countvectorizer_stop_words": "french",
    "countvectorizer_ngram_range": (1, 2),
    "ctfidf_reduce_frequent_words": True,
    "ctfidf_bm25_weighting": False,
    "representation_model": ["MaximalMarginalRelevance"],
    "keybert_nr_repr_docs": 5,
    "keybert_nr_candidate_words": 40,
    "keybert_top_n_words": 20,
    "mmr_diversity": 0.2,
    "mmr_top_n_words": 10,
    "data_language": "English",
}


# Initialize default parameters in Streamlit session state
def initialize_default_parameters_keys():
    for k, v in DEFAULT_PARAMETERS.items():
        if k not in st.session_state:
            st.session_state[k] = v
        register_widget(k)


# Cache data loading function
@st.cache_data
def load_data_wrapper(data_name: Path):
    return load_data(data_name)


# Embedding model options for Streamlit UI
def embedding_model_options():
    return {
        "embedding_model_name": st.selectbox(
            "Name",
            [
                "OrdalieTech/Solon-embeddings-base-0.1",
                "OrdalieTech/Solon-embeddings-large-0.1",
                "dangvantuan/sentence-camembert-large",
                "paraphrase-multilingual-MiniLM-L12-v2",
                "BAAI/bge-base-en-v1.5",
                "sentence-transformers/all-mpnet-base-v2",
                "antoinelouis/biencoder-camembert-base-mmarcoFR",
                "all-MiniLM-L12-v2",
                "all-mpnet-base-v2",
            ],
            key="embedding_model_name",
        ),
        "use_cached_embeddings": st.checkbox(
            "Put embeddings in cache", key="use_cached_embeddings"
        ),
    }


# BERTopic options for Streamlit UI
def bertopic_options():
    return {
        "bertopic_nr_topics": st.number_input(
            "nr_topics",
            min_value=0,
            value=DEFAULT_PARAMETERS["bertopic_nr_topics"],
            key="bertopic_nr_topics",
        ),
        "bertopic_top_n_words": st.number_input(
            "top_n_words",
            min_value=1,
            value=DEFAULT_PARAMETERS["bertopic_top_n_words"],
            key="bertopic_top_n_words",
        ),
    }


# UMAP options for Streamlit UI
def umap_options():
    return {
        "umap_n_neighbors": st.number_input(
            "n_neighbors",
            min_value=1,
            value=DEFAULT_PARAMETERS["umap_n_neighbors"],
            key="umap_n_neighbors",
        ),
        "umap_n_components": st.number_input(
            "n_components",
            min_value=1,
            value=DEFAULT_PARAMETERS["umap_n_components"],
            key="umap_n_components",
        ),
        "umap_min_dist": st.number_input(
            "min_dist",
            min_value=0.0,
            value=DEFAULT_PARAMETERS["umap_min_dist"],
            max_value=1.0,
            key="umap_min_dist",
        ),
        "umap_metric": st.selectbox("metric", ["cosine"], key="umap_metric"),
    }


# HDBSCAN options for Streamlit UI
def hdbscan_options():
    return {
        "hdbscan_min_cluster_size": st.number_input(
            "min_cluster_size",
            min_value=2,
            value=DEFAULT_PARAMETERS["hdbscan_min_cluster_size"],
            key="hdbscan_min_cluster_size",
        ),
        "hdbscan_min_samples": st.number_input(
            "min_samples",
            min_value=1,
            value=DEFAULT_PARAMETERS["hdbscan_min_samples"],
            key="hdbscan_min_samples",
        ),
        "hdbscan_metric": st.selectbox("metric", ["euclidean"], key="hdbscan_metric"),
        "hdbscan_cluster_selection_method": st.selectbox(
            "cluster_selection_method",
            ["eom", "leaf"],
            key="hdbscan_cluster_selection_method",
        ),
        "hdbscan_cluster_selection_epsilon": st.number_input(
            "cluster_selection_epsilon",
            min_value=0.0,
            value=DEFAULT_PARAMETERS["hdbscan_cluster_selection_epsilon"],
            format="%.2f",
            step=0.01,
            key="hdbscan_cluster_selection_epsilon",
        ),
        "hdbscan_max_cluster_size": st.number_input(
            "max_cluster_size",
            min_value=0,
            value=DEFAULT_PARAMETERS["hdbscan_max_cluster_size"],
            key="hdbscan_max_cluster_size",
        ),
        "hdbscan_allow_single_cluster": st.checkbox(
            "allow_single_cluster",
            value=DEFAULT_PARAMETERS["hdbscan_allow_single_cluster"],
            key="hdbscan_allow_single_cluster",
        ),
    }


# CountVectorizer options for Streamlit UI
def countvectorizer_options():
    return {
        "countvectorizer_stop_words": st.selectbox(
            "stop_words", ["french", "english", None], key="countvectorizer_stop_words"
        ),
        "countvectorizer_ngram_range": st.selectbox(
            "ngram_range",
            [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            index=1,
            key="countvectorizer_ngram_range",
        ),
        "countvectorizer_min_df": st.number_input(
            "min_df", min_value=1, value=2, key="countvectorizer_min_df"
        ),
    }


# C-TF-IDF options for Streamlit UI
def ctfidf_options():
    return {
        "ctfidf_reduce_frequent_words": st.checkbox(
            "reduce_frequent_words",
            value=DEFAULT_PARAMETERS["ctfidf_reduce_frequent_words"],
            key="ctfidf_reduce_frequent_words",
        ),
        "ctfidf_bm25_weighting": st.checkbox(
            "bm25_weighting",
            value=DEFAULT_PARAMETERS["ctfidf_bm25_weighting"],
            key="ctfidf_bm25_weighting",
        ),
    }


# Representation model options for Streamlit UI
def representation_model_options():
    options = {}
    available_models = ["KeyBERTInspired", "MaximalMarginalRelevance", "OpenAI"]

    # Model-specific parameters (always visible for ALL models)
    for model in available_models:
        st.write(f"### {model} Parameters")
        if model == "KeyBERTInspired":
            options[f"{model}_nr_repr_docs"] = st.number_input(
                f"{model}: Number of representative documents",
                min_value=1,
                value=5,
                key=f"{model}_nr_repr_docs",
            )
            options[f"{model}_nr_candidate_words"] = st.number_input(
                f"{model}: Number of candidate words",
                min_value=1,
                value=40,
                key=f"{model}_nr_candidate_words",
            )
            options[f"{model}_top_n_words"] = st.number_input(
                f"{model}: Top N words",
                min_value=1,
                value=20,
                key=f"{model}_top_n_words",
            )
        elif model == "MaximalMarginalRelevance":
            options[f"{model}_diversity"] = st.slider(
                f"{model}: Diversity",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                key=f"{model}_diversity",
            )
            options[f"{model}_top_n_words"] = st.number_input(
                f"{model}: Top N words",
                min_value=1,
                value=10,
                key=f"{model}_top_n_words",
            )
        elif model == "OpenAI":
            options[f"{model}_nr_docs"] = st.number_input(
                f"{model}: Number of documents",
                min_value=1,
                value=5,
                key=f"{model}_nr_docs",
            )
            options[f"{model}_language"] = st.selectbox(
                f"{model}: Data Language",
                options=["Français", "English"],
                key=f"{model}_language",
            )

        st.divider()  # Add a divider after each model's parameters, regardless of selection

    st.write("Select representation models:")
    selected_models = st.multiselect(
        "Models",
        options=available_models,
        default=DEFAULT_PARAMETERS["representation_model"],
        key="representation_model",
    )

    # if selected, move OpenAI model to the end of the list
    if "OpenAI" in selected_models:
        selected_models = [model for model in selected_models if model != "OpenAI"] + [
            "OpenAI"
        ]

    options["representation_model"] = selected_models
    return options


@st.cache_data
def plot_topic_treemap(form_parameters, _topic_model, width=700):
    pass


@st.cache_data
def plot_2d_topics(form_parameters, _topic_model, width=700):
    return _topic_model.visualize_topics(width=width)


@st.cache_data
def plot_topics_hierarchy(form_parameters, _topic_model, width=700):
    return _topic_model.visualize_hierarchy(width=width)


def make_dynamic_topics_split(df, nr_bins):
    """
    Split docs into nr_bins and generate a llm_utils timestamp label into a new column
    """
    df = df.sort_values(TIMESTAMP_COLUMN, ascending=False)
    split_df = np.array_split(df, nr_bins)
    for split in split_df:
        split[GROUPED_TIMESTAMP_COLUMN] = split[TIMESTAMP_COLUMN].max()
    return pd.concat(split_df)


@st.cache_data
def compute_topics_over_time(
    form_parameters,
    _topic_model,
    df,
    nr_bins,
    new_df=None,
    new_nr_bins=None,
    new_topics=None,
):
    df = make_dynamic_topics_split(df, nr_bins)
    if new_nr_bins:
        new_df = make_dynamic_topics_split(new_df, new_nr_bins)
        df = pd.concat([df, new_df])
        _topic_model.topics_ += new_topics
    res = _topic_model.topics_over_time(
        df[TEXT_COLUMN],
        df[GROUPED_TIMESTAMP_COLUMN],
        global_tuning=False,
    )
    if new_nr_bins:
        _topic_model.topics_ = _topic_model.topics_[: -len(new_topics)]
    return res


def plot_topics_over_time(
    topics_over_time, dynamic_topics_list, topic_model, time_split=None, width=900
):
    if dynamic_topics_list != "":
        if ":" in dynamic_topics_list:
            dynamic_topics_list = [
                i
                for i in range(
                    int(dynamic_topics_list.split(":")[0]),
                    int(dynamic_topics_list.split(":")[1]),
                )
            ]
        else:
            dynamic_topics_list = [int(i) for i in dynamic_topics_list.split(",")]
        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            topics=dynamic_topics_list,
            width=width,
            title="",
        )
        if time_split:
            fig.add_vline(
                x=time_split,
                line_width=3,
                line_dash="dash",
                line_color="black",
                opacity=1,
            )
        return fig


def print_docs_for_specific_topic(df, topics, topic_number):
    """
    Print documents for a specific topic
    """
    columns_list = [
        col
        for col in [
            TITLE_COLUMN,
            TEXT_COLUMN,
            URL_COLUMN,
            TIMESTAMP_COLUMN,
            CITATION_COUNT_COL,
        ]
        if col in df.keys()
    ]
    df = df.loc[pd.Series(topics) == topic_number][columns_list]
    for _, doc in df.iterrows():
        st.write(f"[{doc.title}]({doc.url})")


@st.cache_data
def transform_new_data(_topic_model, df, embeddings):
    """
    Transform new data using the existing topic model and embeddings.

    Args:
    _topic_model: The trained BERTopic model
    df: DataFrame containing the new data
    embeddings: Pre-computed embeddings for the new data

    Returns:
    Tuple of (topics, probabilities)
    """
    return _topic_model.transform(df[TEXT_COLUMN], embeddings=embeddings)


def plot_docs_reparition_over_time(df, freq):
    """
    Plot document distribution over time
    """
    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime("%Y-%m-%d")

    fig = px.bar(count, x="timestamp", y="size")
    st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)


def plot_remaining_docs_repartition_over_time(df_base, df_remaining, freq):
    """
    Plot remaining document distribution over time
    """
    df = pd.concat([df_base, df_remaining])

    # Get split time value
    split_time = str(df_remaining["timestamp"].min())

    # Print aggregated docs
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime("%Y-%m-%d")
    # Split to set a different color to each DF
    count["category"] = [
        "Base" if time < split_time else "Remaining" for time in count["timestamp"]
    ]

    fig = px.bar(
        count,
        x="timestamp",
        y="size",
        color="category",
        color_discrete_map={
            "Base": "light blue",  # default plotly color to match main page graphs
            "Remaining": "orange",
        },
    )
    st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)


# TODO: Remove "put embeddings in cache" option since it's unadvised due to the large size of embeddings returned by embedding model (sentence and token embeddings)
# TODO: Make the parameters of different representation models appear and disappear based on what was selected in the multi-select box.
