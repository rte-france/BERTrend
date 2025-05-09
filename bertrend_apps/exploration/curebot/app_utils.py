import tomllib
import json
from bertopic import BERTopic
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import plotly.graph_objects as go
from urllib.parse import urlparse

from bertrend import LLM_CONFIG
from bertrend.BERTopicModel import BERTopicModel
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.services.embedding_service import EmbeddingService
from bertrend_apps.exploration.curebot.prompts import (
    TOPIC_DESCRIPTION_SYSTEM_PROMPT,
    TOPIC_SUMMARY_SYSTEM_PROMPT,
)

# Get configuration file
CONFIG = tomllib.load(open(Path(__file__).parent / "config.toml", "rb"))

# Set curebot column name
URL_COLUMN = CONFIG["data"]["url_column"]
TEXT_COLUMN = CONFIG["data"]["text_column"]
TITLE_COLUMN = CONFIG["data"]["title_column"]
SOURCE_COLUMN = CONFIG["data"]["source_column"]
TIMESTAMP_COLUMN = CONFIG["data"]["timestamp_column"]
TAGS_COLUMN = CONFIG["data"]["tags_column"]

# Topics config
TOP_N_WORDS = CONFIG["topics"]["top_n_words"]

# Newsletter
NEWSLETTER_TEMPLATE = CONFIG["newsletter"]["template"]

# Load embdding model
EMBEDDING_SERVICE = EmbeddingService(local=False)


@st.cache_data(show_spinner=False)
def concat_data_from_files(files: list[UploadedFile]) -> pd.DataFrame:
    """
    Concatenate data from multiple Excel files into a single DataFrame.
    """
    df_list = []
    for file in files:
        df_list.append(pd.read_excel(file))
    df = pd.concat(df_list, ignore_index=True)

    return df


@st.cache_data(show_spinner=False)
def chunk_df(
    df: pd.DataFrame, chunk_size: int = 100, overlap: int = 20
) -> pd.DataFrame:
    """
    Split df texts into overlapping chunks while preserving other columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing text and other columns
    chunk_size : int
        Number of words in each chunk
    overlap : int
        Number of words to overlap between chunks

    Returns:
    --------
    pandas.DataFrame
        DataFrame with chunked texts, preserving other column values
    """

    # # Apply chunking to each row
    # chunked_data = []
    # for _, row in df.iterrows():
    #     chunked_data.extend(split_text_to_chunks(row, chunk_size, overlap))

    # # Create new DataFrame from chunked data
    # df_split = (
    #     pd.DataFrame(chunked_data)
    #     .sort_values(by=TIMESTAMP_COLUMN)
    #     .reset_index(drop=True)
    # )

    return df.copy()


@st.cache_data(show_spinner=False)
def get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings for a list of texts."""
    embeddings, _, _ = EMBEDDING_SERVICE.embed(texts)
    return embeddings


@st.cache_data(show_spinner=False)
def fit_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    min_articles_per_topic: int,
    zeroshot_topic_list: list[str] | None = None,
) -> tuple[BERTopic, list[int]]:
    """
    Fit BERTopic model on a list of documents and their embeddings.
    """
    # Override default parameters
    bertopic_config = {"hdbscan_model": {"min_cluster_size": min_articles_per_topic}}
    # Initialize topic model
    topic_model = BERTopicModel(config=bertopic_config)

    # Train topic model
    topic_model_output = topic_model.fit(
        docs=docs,
        embeddings=embeddings,
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=0.65,
    )

    return topic_model_output.topic_model, topic_model_output.topics


@st.cache_data(show_spinner=False)
def get_improved_topic_description(
    df: pd.DataFrame, _topics_info: pd.DataFrame
) -> list[str]:
    """Get improved topic description using LLM."""
    # Get llm client
    llm_client = OpenAI_Client(
        api_key=LLM_CONFIG["api_key"],
        endpoint=LLM_CONFIG["endpoint"],
        model=LLM_CONFIG["model"],
    )

    # List of improved topics description
    improved_descriptions = []

    # Loop over topics
    for topic_number in range(len(_topics_info)):
        topic_df = df[df["topics"] == topic_number]
        user_prompt = "\n\n".join(
            topic_df.apply(
                lambda row: f"Titre : {row[TITLE_COLUMN]}\nArticle : {row[TEXT_COLUMN][0:2000]}...",
                axis=1,
            )
        )
        response = llm_client.generate(
            user_prompt=user_prompt,
            system_prompt=TOPIC_DESCRIPTION_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
        )
        improved_descriptions.append(json.loads(response)["titre"])

    return improved_descriptions


@st.cache_data(show_spinner=False)
def create_newsletter(
    df: pd.DataFrame,
    topics_info: pd.DataFrame,
    nb_topics: int,
    nb_articles_per_topic: int,
):
    """
    Create newsletter dict from containing the newsletter content in the following format:
    {
        "title": "Newsletter",
        "min_timestamp": "Monday 01 January 2025",
        "max_timestamp": "Sunday 07 January 2025",
        "topics": [
            {
                "title": "Topic 1",
                "keywords": "#keyword1 #keyword2 #keyword3",
                "summary": "Summary of topic 1"
                "articles": [
                    {
                        "title": "Article 1",
                        "url": "https://www.article1.com",
                        "timestamp": "Monday 01 January 2023",
                        "source": "Source 1"
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """
    # Create newsletter dict that stores the newsletter content
    newsletter_dict = {"title": "Newsletter"}

    # Get min and max date of the articles in the dataframe
    newsletter_dict["min_timestamp"] = (
        df[TIMESTAMP_COLUMN].min().strftime("%A %d %B %Y")
    )
    newsletter_dict["max_timestamp"] = (
        df[TIMESTAMP_COLUMN].max().strftime("%A %d %B %Y")
    )

    newsletter_dict["topics"] = []
    for i in range(nb_topics):
        # Dict to store topic info
        topic_dict = {}

        # Get title and key words
        topic_dict["title"] = topics_info.iloc[i]["llm_description"]
        topic_dict["keywords"] = (
            "#" + " #".join(topics_info.iloc[i]["Representation"]).strip()
        )

        # Filter df to get articles for the topic
        topic_df = df[df["topics"] == i]

        # Get first `newsletter_nb_articles_per_topic` articles for the topic
        topic_df = topic_df.head(min(nb_articles_per_topic, len(topic_df)))

        # Get a summary of the topic
        user_prompt = "\n\n".join(
            topic_df.apply(
                lambda row: f"Titre : {row[TITLE_COLUMN]}\nArticle : {row[TEXT_COLUMN][0:2000]}...",
                axis=1,
            )
        )
        llm_client = OpenAI_Client(
            api_key=LLM_CONFIG["api_key"],
            endpoint=LLM_CONFIG["endpoint"],
            model=LLM_CONFIG["model"],
        )
        response = llm_client.generate(
            user_prompt=user_prompt,
            system_prompt=TOPIC_SUMMARY_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
        )
        topic_dict["summary"] = json.loads(response)["résumé"]
        topic_dict["articles"] = []
        for _, row in topic_df.iterrows():
            article_dict = {}
            article_dict["title"] = row[TITLE_COLUMN]
            article_dict["url"] = row[URL_COLUMN]
            article_dict["timestamp"] = row[TIMESTAMP_COLUMN].strftime("%A %d %B %Y")
            article_dict["source"] = row[SOURCE_COLUMN]
            topic_dict["articles"].append(article_dict)
        newsletter_dict["topics"].append(topic_dict)
    return newsletter_dict


def display_source_distribution(
    representative_df: pd.DataFrame, selected_sources: list[str]
):
    """Display the distribution of sources in a pie chart."""

    source_counts = representative_df[URL_COLUMN].apply(get_website_name).value_counts()

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

    st.plotly_chart(fig, use_container_width=True)


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


def display_representative_documents(filtered_df: pd.DataFrame):
    """Display representative documents for the selected topic."""
    with st.container(border=False, height=600):
        for _, doc in filtered_df.iterrows():
            website_name = get_website_name(doc[URL_COLUMN])
            date = doc[TIMESTAMP_COLUMN].strftime("%A %d %b %Y %H:%M:%S")
            snippet = (
                doc[TEXT_COLUMN][:200] + "..."
                if len(doc[TEXT_COLUMN]) > 200
                else doc[TEXT_COLUMN]
            )

            content = f"""**{doc[TITLE_COLUMN]}**\n\n{date} | {'Unknown Source' if website_name == 'Unknown Source' else website_name}\n\n{snippet}"""

            if website_name != "Unknown Source":
                st.link_button(content, doc[URL_COLUMN])
            else:
                st.markdown(content)
