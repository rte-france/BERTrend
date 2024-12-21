#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import locale
from pathlib import Path

import datamapplot
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from loguru import logger
from umap import UMAP

from bertrend.demos.topic_analysis.app_utils import (
    plot_2d_topics,
)
from bertrend.demos.topic_analysis.state_utils import restore_widget_state
from bertrend.demos.weak_signals.visualizations_utils import PLOTLY_BUTTON_SAVE_CONFIG
from bertrend.utils.data_loading import TEXT_COLUMN

# Set locale for French date names
locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

# Restore widget state and set up page
restore_widget_state()
st.set_page_config(page_title="BERTrend topic analysis", layout="wide")

# Check if a model is trained
if "topic_model" not in st.session_state:
    st.error("Train a model to explore different visualizations.", icon="🚨")
    st.stop()


@st.cache_data
def overall_results():
    """Display overall results visualization."""
    try:
        return plot_2d_topics(
            st.session_state.parameters, st.session_state["topic_model"]
        )
    except TypeError as te:
        logger.error(f"Error occurred: {te}")
        return None
    except ValueError as ve:
        logger.error(f"Error occurred: {ve}")
        return None


@st.cache_data
def create_topic_info_dataframe():
    """Create a DataFrame containing topics, number of documents per topic, and list of documents for each topic."""
    docs = st.session_state["timefiltered_df"][TEXT_COLUMN].tolist()
    topic_assignments = st.session_state["topics"]

    topic_info = pd.DataFrame({"Document": docs, "Topic": topic_assignments})
    topic_info_agg = (
        topic_info.groupby("Topic").agg({"Document": ["count", list]}).reset_index()
    )

    topic_info_agg.columns = ["topic", "number_of_documents", "list_of_documents"]
    topic_info_agg["topic"] = topic_info_agg["topic"].apply(
        lambda x: (
            ", ".join(
                [word for word, _ in st.session_state["topic_model"].get_topic(x)]
            )
            if x != -1
            else "Outlier"
        )
    )

    return topic_info_agg[topic_info_agg["topic"] != "Outlier"]


@st.cache_data
def create_treemap():
    """Create a treemap visualization of topics and their corresponding documents."""
    topic_info_df = create_topic_info_dataframe()

    labels, parents, values = [], [], []
    for _, row in topic_info_df.iterrows():
        topic_label = f"{row['topic']} ({row['number_of_documents']})"
        labels.append(topic_label)
        parents.append("")
        values.append(row["number_of_documents"])

        for doc in row["list_of_documents"]:
            labels.append(doc[:50] + "...")
            parents.append(topic_label)
            values.append(1)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            textinfo="label+value",
            marker=dict(colors=[], line=dict(width=0), pad=dict(t=0)),
            textfont=dict(size=34),
        )
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.update_traces(marker=dict(cornerradius=10))
    return fig


def get_binary_file_downloader_html(bin_file, file_label="File"):
    with open(bin_file, "rb") as f:
        data = f.read()
    return data


def calculate_document_lengths(documents):
    """Calculate the length of each document in terms of word count."""
    return np.array([len(doc.split()) for doc in documents])


@st.cache_data
def create_datamap(include_outliers):
    """Create an interactive data map visualization."""
    reduced_embeddings = UMAP(
        n_neighbors=5, n_components=2, min_dist=0.2, metric="cosine"
    ).fit_transform(st.session_state["embeddings"])

    topic_nums = list(set(st.session_state["topics"]))
    topic_info = st.session_state["topic_model"].get_topic_info()
    topic_representations = {
        row["Topic"]: row["Name"]
        for _, row in topic_info.iterrows()
        if row["Topic"] in topic_nums
    }
    docs = st.session_state["timefiltered_df"][TEXT_COLUMN].tolist()

    df = pd.DataFrame(
        {
            "document": docs,
            "embedding": list(reduced_embeddings),
            "topic_num": st.session_state["topics"],
            "topic_representation": [
                topic_representations.get(topic, f"-1_{topic}")
                for topic in st.session_state["topics"]
            ],
        }
    )

    df["is_noise"] = df["topic_representation"].str.contains("-1")
    df["topic_representation"] = df.apply(
        lambda row: (
            "" if row["is_noise"] else topic_representations.get(row["topic_num"], "")
        ),
        axis=1,
    )
    df["topic_color"] = df.apply(
        lambda row: "#999999" if row["is_noise"] else None, axis=1
    )

    if not include_outliers:
        df = df[~df["is_noise"]]

    if df.empty:
        return None

    embeddings_array = np.array(df["embedding"].tolist())
    topic_representations_array = df["topic_representation"].values

    unique_topics = df["topic_representation"].unique()
    unique_topics = unique_topics[unique_topics != ""]
    color_palette = sns.color_palette("tab20", len(unique_topics)).as_hex()
    color_mapping = dict(zip(unique_topics, color_palette))
    df["topic_color"] = df.apply(
        lambda row: color_mapping.get(row["topic_representation"], row["topic_color"]),
        axis=1,
    )

    try:
        hover_data = df["document"].tolist()
        document_lengths = calculate_document_lengths(hover_data)
        normalized_sizes = 5 + 45 * (document_lengths - document_lengths.min()) / (
            document_lengths.max() - document_lengths.min()
        )

        plot = datamapplot.create_interactive_plot(
            embeddings_array,
            topic_representations_array,
            hover_text=hover_data,
            marker_size_array=normalized_sizes,
            inline_data=True,
            noise_label="",
            noise_color="#999999",
            color_label_text=True,
            label_wrap_width=16,
            label_color_map=color_mapping,
            width="100%",
            height="100%",
            darkmode=False,
            marker_color_array=df["topic_color"].values,
            use_medoids=True,
            cluster_boundary_polygons=False,
            enable_search=True,
            search_field="hover_text",
            point_line_width=0,
            logo="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/RTE_logo.svg/1024px-RTE_logo.svg.png",
            logo_width=100,
        )

        save_path = Path(__file__).parent.parent / "datamapplot.html"
        with open(save_path, "wb") as f:
            f.write(plot._html_str.encode(encoding="UTF-8", errors="replace"))

        with open(save_path, "r", encoding="utf-8") as HtmlFile:
            source_code = HtmlFile.read()
        return source_code

    except Exception as e:
        logger.error(f"Error in creating datamap: {str(e)}")
        return None


# Main execution
st.title("Visualizations")

# Sidebar
with st.sidebar:
    include_outliers = st.checkbox("Include outliers (Topic = -1)", value=True)

# Overall Results
with st.expander("Overall Results", expanded=False):
    overall_results_plot = overall_results()
    if overall_results_plot is not None:
        st.plotly_chart(
            overall_results_plot,
            config=PLOTLY_BUTTON_SAVE_CONFIG,
            use_container_width=True,
        )
    else:
        st.error("Cannot display overall results", icon="🚨")
        st.warning("Try to change the UMAP parameters", icon="⚠️")

# Topics Treemap
with st.expander("Topics Treemap", expanded=False):
    with st.spinner("Computing topics treemap..."):
        treemap_plot = create_treemap()
        st.plotly_chart(
            treemap_plot, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True
        )


# Data Map
with st.expander("Data Map", expanded=True):
    with st.spinner("Loading Data-map plot..."):
        datamap_html = create_datamap(include_outliers)
        if datamap_html is not None:
            # Using st.html does fix the width and height issue by making it adaptive, but the html never loads (it loops indefinitely)
            # This is the best solution so far, with a button to save the html and view it in fullscreen later.
            components.html(datamap_html, width=1200, height=1000, scrolling=True)

            # Add the fullscreen button
            save_path = Path(__file__).parent.parent / "datamapplot.html"

            # Create a download button
            st.download_button(
                label="View in fullscreen",
                data=get_binary_file_downloader_html(save_path),
                file_name="datamapplot.html",
                mime="text/html",
                use_container_width=True,
                type="secondary",
            )
        else:
            st.warning(
                "No valid topics to visualize. All documents might be classified as outliers."
            )


# FIXME: cluster_boundary_polygons=True causes a "pop from an empty set" error in the data map plot's generation process.
# It's not urgent, but should be looked into to see what's causing the problem and potentially get a better visualization where clusters are delimitted with contours.
