#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import inspect
import pandas as pd
import streamlit as st
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from bertrend.services.summary.chatgpt_summarizer import GPTSummarizer
from bertrend.train import train_BERTopic
from bertrend.utils.data_loading import (
    load_data,
    TIMESTAMP_COLUMN,
    split_df_by_paragraphs,
    clean_dataset,
)

from bertrend_apps.data_provider.curebot_provider import CurebotProvider
from bertrend_apps.newsletters.newsletter_features import generate_newsletter, md2html

COLUMN_URL = "url"
MIN_TEXT_LENGTH = 150
EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-large"
TOP_N_WORDS = 5
# EMBEDDING_MODEL_NAME = "antoinelouis/biencoder-camembert-base-mmarcoFR"

css_style = Path(inspect.getfile(generate_newsletter)).parent / "newsletter.css"

if "topic_detection_disabled" not in st.session_state:
    st.session_state.topic_detection_disabled = False
if "newsletter_disabled" not in st.session_state:
    st.session_state.newsletter_disabled = False
if "import_expanded" not in st.session_state:
    st.session_state.import_expanded = True
if "st.session_state.topic_expanded" not in st.session_state:
    st.session_state.topic_expanded = True


@st.cache_data
def parse_data_from_files(files: List[UploadedFile]) -> pd.DataFrame:
    """Read a list of excel files and return a single dataframe containing the data"""
    dataframes = []

    with TemporaryDirectory() as tmpdir:
        for f in files:
            with open(tmpdir + "/" + f.name, "wb") as tmp_file:
                tmp_file.write(f.getvalue())
                print(tmp_file.name)

            if tmp_file is not None:
                with st.spinner(f"Analyse des articles de: {f.name}"):
                    provider = CurebotProvider(curebot_export_file=Path(tmp_file.name))
                    articles = provider.get_articles()
                    articles_path = Path(tmpdir) / (f.name + ".jsonl")
                    provider.store_articles(articles, articles_path)
                    df = (
                        load_data(articles_path)
                        .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
                        .reset_index(drop=True)
                        .reset_index()
                    )
                    dataframes.append(df)

        # Concat all dataframes
        df_concat = pd.concat(dataframes, ignore_index=True)
        df_concat = df_concat.drop_duplicates(subset=COLUMN_URL, keep="first")
        return df_concat


@st.cache_data
def parse_data_from_feed(feed_url):
    """Return a single dataframe containing the data obtained from the feed"""
    with TemporaryDirectory() as tmpdir:
        with st.spinner(f"Analyse des articles de: {feed_url}"):
            provider = CurebotProvider(feed_url=feed_url)
            articles = provider.get_articles()
            articles_path = Path(tmpdir) / "feed.jsonl"
            provider.store_articles(articles, articles_path)
            return (
                load_data(articles_path)
                .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
                .reset_index(drop=True)
                .reset_index()
            )


def split_data():
    st.session_state["df_split"] = (
        split_df_by_paragraphs(st.session_state["df"])
        .drop("index", axis=1)
        .sort_values(
            by=TIMESTAMP_COLUMN,
            ascending=False,
        )
        .reset_index(drop=True)
        .reset_index()
    )

    # Clean dataset using min_text_length
    st.session_state["df_split"] = clean_dataset(
        st.session_state["df_split"],
        MIN_TEXT_LENGTH,
    )


def train_model():
    (
        st.session_state["topic_model"],
        st.session_state["topics"],
        _,
        _,
        _,
        _,
    ) = train_BERTopic(
        full_dataset=st.session_state["df_split"],
        indices=None,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        use_cache=False,
        top_n_words=TOP_N_WORDS,
    )


def create_newsletter():
    with st.spinner("Création de la newsletters..."):
        st.session_state["newsletters"], _, _ = generate_newsletter(
            topic_model=st.session_state["topic_model"],
            df=st.session_state["df"],
            df_split=st.session_state["df_split"],
            topics=st.session_state["topics"],
            top_n_topics=st.session_state["newsletter_nb_topics"],
            top_n_docs=st.session_state["newsletter_nb_docs"],
            improve_topic_description=True,
            summarizer_class=GPTSummarizer,
            summary_mode="topic",
            openai_model_name=st.session_state["openai_model_name"],
            nb_sentences=st.session_state["nb_sentences"],
        )


@st.experimental_dialog("Newsletter preview", width="large")
def preview_newsletter():
    content = md2html(st.session_state["final_newsletter"], css_style=css_style)
    st.html(content)


def import_data():
    with st.expander(
        "**Import des données de Curebot**", expanded=st.session_state.import_expanded
    ):
        # uploader
        st.text_input(
            "URL du flux ATOM / RSS",
            value=(
                st.session_state["curebot_rss_url"]
                if "curebot_rss_url" in st.session_state
                else ""
            ),
            key="curebot_rss_url",
            help="Saisir le l'URL complète du flux Curebot à importer (par ex. https://api-a1.beta.curebot.io/v1/atom-feed/smartfolder/a5b14e159caa4cb5967f94e84640f602)",
        )
        uploaded_files = st.file_uploader(
            "Fichiers Excel (format Curebot .xlsx)",
            accept_multiple_files=True,
            help="Glisser/déposer dans cette zone les exports Curebot au format Excel",
        )

    # check content
    if uploaded_files or "curebot_rss_url" in st.session_state:
        if uploaded_files:
            st.session_state["df"] = parse_data_from_files(uploaded_files)
        elif st.session_state["curebot_rss_url"]:
            st.session_state["df"] = parse_data_from_feed(
                st.session_state["curebot_rss_url"]
            )

        # split and clean data
        if "df" in st.session_state:
            split_data()
            logger.info(f"Size of dataset: {len(st.session_state['df_split'])}")


def display_data():
    if "df" in st.session_state:
        st.session_state.import_expanded = False
        # Affichage du contenu du fichier Excel
        with st.expander("**Contenu des données**", expanded=False):
            st.dataframe(st.session_state["df"])


def detect_topics():
    if "df_split" in st.session_state:
        st.session_state.import_expanded = False
        with st.expander(
            "**Détection de topics**", expanded=st.session_state.topic_expanded
        ):
            # Topic detection
            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "Détection des topics",
                    on_click=train_model,
                    key="topic_detection",
                    type="primary",
                    disabled=st.session_state.topic_detection_disabled,
                )
            with col2:
                if "topic_model" in st.session_state:
                    st.info(
                        f"Nombre de topics: {len(st.session_state['topic_model'].get_topic_info()) - 1}"
                    )


def newsletter_creation():
    # Newsletter creation
    if "topic_model" in st.session_state.keys():
        st.session_state.topic_expanded = False
        with st.expander("**Création de la newsletters**", expanded=True):
            # st.session_state.topic_detection_disabled = True
            generation_button = st.button(
                "Génération de newsletters",
                on_click=create_newsletter,
                type="primary",
                disabled=st.session_state.newsletter_disabled,
            )

            # Edit manually newsletters
            if "newsletters" in st.session_state.keys():
                st.text_area(
                    "Contenu éditable de la newsletters (faire CTRL+ENTREE pour prendre en compte les modifications)",
                    value=(
                        st.session_state["newsletters"]
                        if (
                            "final_newsletter" not in st.session_state
                            or generation_button
                        )
                        else st.session_state["final_newsletter"]
                    ),
                    height=400,
                    key="final_newsletter",
                )

                if "final_newsletter" in st.session_state:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Preview", on_click=preview_newsletter)
                    with col2:
                        # Newsletter download
                        st.download_button(
                            "Téléchargement",
                            file_name="newsletters.html",
                            data=md2html(
                                st.session_state["final_newsletter"],
                                css_style=css_style,
                            ),
                            type="primary",
                        )


def main_page():
    """Main page rendering"""
    # title
    st.title("Wattelse Veille & Analyse")
    import_data()
    display_data()
    detect_topics()
    newsletter_creation()


def options():
    with st.sidebar:
        st.title("Réglages")

        st.slider(
            "Nombre max de topics",
            min_value=1,
            max_value=10,
            value=5,
            key="newsletter_nb_topics",
        )

        st.slider(
            "Nombre max d'articles par topics",
            min_value=1,
            max_value=10,
            value=5,
            key="newsletter_nb_docs",
        )

        st.slider(
            "Longueur des synthèses (# phrases)",
            min_value=1,
            max_value=10,
            value=4,
            key="nb_sentences",
        )

        st.selectbox(
            "Moteur de résumé",
            ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
            key="openai_model_name",
        )


def main():
    options()
    main_page()


# Main
main()
