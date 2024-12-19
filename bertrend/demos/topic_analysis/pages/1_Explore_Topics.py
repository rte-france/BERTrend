#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import io
import locale
import os
import re
import zipfile

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from bertrend import LLM_CONFIG
from bertrend.demos.topic_analysis.state_utils import restore_widget_state
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.trend_analysis.visualizations import PLOTLY_BUTTON_SAVE_CONFIG
from bertrend.utils.data_loading import TIMESTAMP_COLUMN, TEXT_COLUMN
from bertrend_apps.newsletters.newsletter_features import get_most_representative_docs
from bertrend.demos.topic_analysis.app_utils import (
    compute_topics_over_time,
    plot_topics_over_time,
    print_docs_for_specific_topic,
)


def generate_topic_description(topic_model, topic_number, filtered_docs):
    # Ensure OpenAI API key is set in the environment variables
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OpenAI API key not found in environment variables")
        return "Error: OpenAI API key not set"

    # Get topic representation
    topic_words = topic_model.get_topic(topic_number)
    topic_representation = ", ".join(
        [word for word, _ in topic_words[:10]]
    )  # Get top 10 words

    # Prepare the documents text
    docs_text = "\n\n".join(
        [
            f"Document {i + 1}: {doc.text}..."
            for i, doc in filtered_docs.head(3).iterrows()
        ]
    )

    # Prepare the prompt
    prompt = f"""En tant qu'expert en analyse thématique, votre tâche est de générer un titre et une description pour un thème spécifique.


    Représentation du thème : {topic_representation}

    Voici le contexte de ce thème :

    {docs_text}

    Basé sur ces informations, veuillez fournir :
    1. Un titre concis et informatif pour ce thème (maximum 10 mots)
    2. Une description détaillée du thème (environ 100 mots)

    Format de réponse :
    ### Titre : [Votre titre ici]
    [Votre description ici]
    """

    # logger.debug(f"Prompt for GPT:\n{prompt}")

    try:
        client = OpenAI_Client(
            api_key=LLM_CONFIG["api_key"],
            endpoint=LLM_CONFIG["endpoint"],
            model=LLM_CONFIG["model"],
        )
        return client.generate(user_prompt=prompt)
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return f"Error generating description: {str(e)}"


# Set locale for French date names
locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

# Constants
EXPORT_BASE_FOLDER = Path(__file__).parent.parent / "exported_topics"

# Restore widget state
restore_widget_state()


def check_model_and_prepare_topics():
    """
    Check if a model is trained and prepare topics over time if necessary.
    """
    if "topic_model" not in st.session_state:
        st.error("Train a model to explore generated topics.", icon="🚨")
        st.stop()

    if (
        "topics_over_time" not in st.session_state
        and TIMESTAMP_COLUMN in st.session_state["timefiltered_df"]
    ):
        st.session_state["topics_over_time"] = compute_topics_over_time(
            st.session_state["parameters"],
            st.session_state["topic_model"],
            st.session_state["timefiltered_df"],
            nr_bins=10,
        )


def set_topic_selection(selected_topic_number):
    """Set the selected topic number in the session state."""
    st.session_state["selected_topic_number"] = selected_topic_number


def find_similar_topic():
    """Find a topic similar to the search terms."""
    similar_topics, _ = st.session_state["topic_model"].find_topics(
        st.session_state["search_terms"], top_n=1
    )
    st.session_state["selected_topic_number"] = similar_topics[0]


def display_sidebar():
    """Display the sidebar with search functionality and topic list."""
    with st.sidebar:
        # Search bar
        search_terms = st.text_input(
            "Search topic", on_change=find_similar_topic, key="search_terms"
        )

        # Topics list
        for index, topic in st.session_state["topics_info"].iterrows():
            topic_number = topic["Topic"]
            topic_words = topic["Representation"][:3]
            button_title = f"{topic_number} - {' | '.join(topic_words)}"
            if "new_topics" in st.session_state:
                new_docs_number = st.session_state["new_topics"].count(topic_number)
                button_title += f" :red[+{new_docs_number}]"
            st.button(
                button_title,
                use_container_width=True,
                on_click=set_topic_selection,
                args=(topic_number,),
            )


def display_topic_info():
    """Display the main topic information."""
    topic_docs_number = st.session_state["topics_info"].iloc[
        st.session_state["selected_topic_number"]
    ]["Count"]
    topic_words = st.session_state["topics_info"].iloc[
        st.session_state["selected_topic_number"]
    ]["Representation"]

    st.write(
        f"# Topic {st.session_state['selected_topic_number']} : {topic_docs_number} documents"
    )
    st.markdown(f"## #{' #'.join(topic_words)}")


def plot_topic_over_time():
    """Plot the topic evolution over time."""
    if TIMESTAMP_COLUMN in st.session_state["timefiltered_df"]:
        if "new_topics_over_time" not in st.session_state:
            st.plotly_chart(
                plot_topics_over_time(
                    st.session_state["topics_over_time"],
                    str(st.session_state["selected_topic_number"]),
                    st.session_state["topic_model"],
                ),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                plot_topics_over_time(
                    st.session_state["new_topics_over_time"],
                    str(st.session_state["selected_topic_number"]),
                    st.session_state["topic_model"],
                    time_split=st.session_state["timefiltered_df"][
                        TIMESTAMP_COLUMN
                    ].max(),
                ),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                use_container_width=True,
            )


def get_representative_documents(top_n_docs):
    """Get representative documents for the selected topic."""
    if st.session_state["split_by_paragraphs"]:
        return get_most_representative_docs(
            st.session_state["topic_model"],
            st.session_state["initial_df"],
            st.session_state["topics"],
            df_split=st.session_state["timefiltered_df"],
            topic_number=st.session_state["selected_topic_number"],
            top_n_docs=top_n_docs,
        )
    else:
        docs_by_cluster_probability = get_most_representative_docs(
            st.session_state["topic_model"],
            st.session_state["timefiltered_df"],
            st.session_state["topics"],
            topic_number=st.session_state["selected_topic_number"],
            mode="cluster_probability",
            top_n_docs=top_n_docs,
        )
        docs_by_ctfidf_similarity = get_most_representative_docs(
            st.session_state["topic_model"],
            st.session_state["timefiltered_df"],
            st.session_state["topics"],
            topic_number=st.session_state["selected_topic_number"],
            mode="ctfidf_representation",
            top_n_docs=top_n_docs,
        )
        return pd.concat(
            [docs_by_cluster_probability, docs_by_ctfidf_similarity]
        ).drop_duplicates()


def display_source_distribution(representative_df, selected_sources):
    """Display the distribution of sources in a pie chart."""

    source_counts = representative_df["url"].apply(get_website_name).value_counts()

    # Create a list to store the 'pull' values for each slice
    pull = []

    # Determine which slices should be pulled out
    for source in source_counts.index:
        if source in selected_sources and "All" not in selected_sources:
            pull.append(0.2)
        else:
            pull.append(0)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=source_counts.index,
                values=source_counts.values,
                pull=pull,
                textposition="inside",
                textinfo="percent+label",
                hole=0.3,
            )
        ]
    )

    fig.update_layout(
        showlegend=False, height=600, width=500, margin=dict(t=0, b=0, l=0, r=0)
    )

    st.plotly_chart(fig, config=PLOTLY_BUTTON_SAVE_CONFIG, use_container_width=True)


def get_website_name(url):
    """Extract website name from URL, handling None, NaN, and invalid URLs."""
    if pd.isna(url) or url is None or isinstance(url, float):
        return "Unknown Source"
    try:
        return (
            urlparse(str(url)).netloc.replace("www.", "").split(".")[0]
            or "Unknown Source"
        )
    except:
        return "Unknown Source"


def display_representative_documents(filtered_df):
    """Display representative documents for the selected topic."""
    with st.container(border=False, height=600):
        for _, doc in filtered_df.iterrows():
            website_name = get_website_name(doc.url)
            date = doc.timestamp.strftime("%A %d %b %Y %H:%M:%S")
            snippet = doc.text[:200] + "..." if len(doc.text) > 150 else doc.text

            content = f"""**{doc.title}**\n\n{date} | {'Unknown Source' if website_name == 'Unknown Source' else website_name}\n\n{snippet}"""

            if website_name != "Unknown Source":
                st.link_button(content, doc.url)
            else:
                st.markdown(content)
            st.divider()


def display_new_documents():
    """Display new documents if remaining data is processed."""
    if "new_topics_over_time" in st.session_state:
        st.write("## New documents")
        print_docs_for_specific_topic(
            st.session_state["remaining_df"],
            st.session_state["new_topics"],
            st.session_state["selected_topic_number"],
        )


def create_topic_documents(
    filtered_df, topic_model, granularity_days, TIMESTAMP_COLUMN, TEXT_COLUMN
):
    """Create topic documents grouped by a given time granularity."""
    topic_docs = filtered_df.sort_values(by=TIMESTAMP_COLUMN)
    documents = []

    if not topic_docs.empty:
        topic_number = st.session_state["selected_topic_number"]
        topic_words = [word for word, _ in topic_model.get_topic(topic_number)]

        folder_name = (
            f"topic_{topic_number}_{' '.join(topic_words[:3])}_{granularity_days}days"
        )
        folder_name = re.sub(r"[^\w\-_\. ]", "_", folder_name)

        start_date = topic_docs[TIMESTAMP_COLUMN].min()
        end_date = topic_docs[TIMESTAMP_COLUMN].max()
        current_date = start_date

        while current_date <= end_date:
            next_date = current_date + timedelta(days=granularity_days)
            period_docs = topic_docs[
                (topic_docs[TIMESTAMP_COLUMN] >= current_date)
                & (topic_docs[TIMESTAMP_COLUMN] < next_date)
            ]

            if not period_docs.empty:
                file_name = f"{current_date.strftime('%Y%m%d')}-{(next_date - timedelta(days=1)).strftime('%Y%m%d')}.txt"
                content = "\n\n".join(period_docs[TEXT_COLUMN])
                documents.append((file_name, content))

            current_date = next_date

    return folder_name, documents


def main():
    """Main function to run the Streamlit topic_analysis."""
    check_model_and_prepare_topics()
    display_sidebar()

    if "selected_topic_number" not in st.session_state:
        st.stop()

    display_topic_info()
    plot_topic_over_time()

    st.divider()

    # Number of articles to display
    top_n_docs = st.number_input(
        "Number of articles to display",
        min_value=1,
        max_value=st.session_state["topics_info"].iloc[
            st.session_state["selected_topic_number"]
        ]["Count"],
        value=st.session_state["topics_info"].iloc[
            st.session_state["selected_topic_number"]
        ]["Count"],
        step=1,
    )

    representative_df = get_representative_documents(top_n_docs)
    representative_df = representative_df.sort_values(by="timestamp", ascending=False)

    # Get unique sources
    sources = representative_df["url"].apply(get_website_name).unique()

    # Multi-select for sources
    selected_sources = st.multiselect(
        "Select the sources to display",
        options=["All"] + list(sources),
        default=["All"],
    )

    # Filter the dataframe based on selected sources
    if "All" not in selected_sources:
        filtered_df = representative_df[
            representative_df["url"].apply(get_website_name).isin(selected_sources)
        ]
    else:
        filtered_df = representative_df

    # Create two columns
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        # Pass the full representative_df to display_source_distribution
        display_source_distribution(representative_df, selected_sources)

    with col2:
        # Filter the dataframe only for document display
        if "All" not in selected_sources:
            filtered_df = representative_df[
                representative_df["url"].apply(get_website_name).isin(selected_sources)
            ]
        else:
            filtered_df = representative_df
        display_representative_documents(filtered_df)

    display_new_documents()

    st.divider()

    # GPT description button
    if st.button(
        "Generate a short description of the topic",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Génération de la description en cours..."):
            gpt_description = generate_topic_description(
                st.session_state["topic_model"],
                st.session_state["selected_topic_number"],
                filtered_df,
            )
        with st.container(border=True):
            st.markdown(gpt_description)

    st.divider()

    # Export configuration
    st.subheader("Export Configuration")
    export_method = st.radio(
        "Choose export method:",
        ("Download as ZIP", "Save to folder"),
        index=0,
        help="Select whether to download documents as a ZIP file or save them directly to a folder on the server.",
    )

    granularity_days = st.number_input(
        "Granularity (number of days)", min_value=1, value=3, step=1
    )

    if export_method == "Download as ZIP":
        # Prepare ZIP file
        folder_name, documents = create_topic_documents(
            filtered_df,
            st.session_state["topic_model"],
            granularity_days,
            TIMESTAMP_COLUMN,
            TEXT_COLUMN,
        )

        if documents:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zip_file:
                for file_name, content in documents:
                    zip_file.writestr(file_name, content)

            zip_buffer.seek(0)
            zip_filename = f"{folder_name}.zip"

            st.download_button(
                label="Export Topic Documents",
                data=zip_buffer,
                file_name=zip_filename,
                mime="application/zip",
            )
        else:
            st.warning("No documents found for the selected topic.")
    else:
        if st.button("Export Topic Documents"):
            folder_name, documents = create_topic_documents(
                filtered_df,
                st.session_state["topic_model"],
                granularity_days,
                TIMESTAMP_COLUMN,
                TEXT_COLUMN,
            )

            if documents:
                export_folder = EXPORT_BASE_FOLDER / folder_name
                export_folder.mkdir(parents=True, exist_ok=True)

                for file_name, content in documents:
                    file_path = export_folder / file_name
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                st.success(
                    f"Successfully exported documents to folder: {export_folder}"
                )
            else:
                st.warning("No documents found for the selected topic.")


if __name__ == "__main__":
    main()

# FIXME: The number of documents being displayed per topic corresponds to the paragraphs, it should instead correspond to the number of original articles before splitting.
# TODO: Granularity to export multiple .txt files in order to use them later with Text2KG hasn't been properly checked (Scenario where granularity is bigger than number of days a topic spans over, or bigger than half... etc)
