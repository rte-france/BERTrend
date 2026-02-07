#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd
import streamlit as st
from loguru import logger

from bertrend.bertrend_apps.exploration.curebot.app_utils import (
    TAGS_COLUMN,
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    TITLE_COLUMN,
    TOP_N_WORDS,
    concat_data_from_files,
    fit_bertopic,
    get_embeddings,
    get_improved_topic_description,
)


def show() -> None:
    # Data uploading component
    upload_data()

    # Load data into dataframe
    if st.session_state.get("uploaded_files"):
        try:
            preprocess_data()
        except Exception as e:
            logger.error(e)
            st.error(
                f"Erreur lors du chargement des données. Vérifiez que vos données respectent le format Curebot attendu."
            )

    # If data is loaded
    if "df" in st.session_state:
        # Show data
        with st.expander("Voir les données", expanded=False):
            st.write(st.session_state["df"])

        # Button to train model
        if st.button("Détecter les sujets", type="primary"):
            with st.spinner("Détection des sujets..."):
                # Train model
                train_model()

        # If topic model is trained, update df with topics and llm description
        if "topic_model" in st.session_state:
            st.session_state["df"]["topics"] = st.session_state["topics"]
            with st.spinner("Génération des titres des sujets..."):
                st.session_state["topics_info"]["llm_description"] = (
                    get_improved_topic_description(
                        st.session_state["df"], st.session_state["topics_info"]
                    )
                )
            st.success("Sujets détectés, voir l'onglet résultats.")


def upload_data() -> None:
    """
    Data uploading component for Curebot format.
    Sets in sessions_state:
      - "uploaded_files": list of uploaded files
    """
    # Excel files input
    st.session_state["uploaded_files"] = st.file_uploader(
        "Fichiers Excel au format rapport Curebot `.xlsx`",
        accept_multiple_files=True,
        help="Glisser/déposer dans cette zone les exports Curebot au format Excel",
    )


def preprocess_data() -> None:
    """
    Preprocess data from uploaded files.
    Sets in session_state:
      - "df": dataframe with all data
    """
    # Concatenate uploaded Excel files into a single dataframe
    df = concat_data_from_files(st.session_state["uploaded_files"])

    # Remove duplicates based on title and text columns
    df = df.drop_duplicates(subset=[TITLE_COLUMN, TEXT_COLUMN]).reset_index(drop=True)

    # Remove rows where text is empty
    df = df[df[TEXT_COLUMN].notna()].reset_index(drop=True)

    # Sort df by date
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df = df.sort_values(by=TIMESTAMP_COLUMN, ascending=False).reset_index(drop=True)

    st.session_state["df"] = df


def train_model() -> None:
    """
    Train a BERTopic model based on provided data.
    Sets in session_state:
      - "embeddings": embeddings of the dataset
      - "topic_model": trained BERTopic model
      - "topics": topics extracted by the model
      - "topics_info": information about the topics
    """
    # Get texts list and embeddings
    texts_list = st.session_state["df"][TEXT_COLUMN].tolist()
    embeddings = get_embeddings(texts_list)

    # If use_tags is True, get tags from dataframe
    if st.session_state["use_tags"]:
        # Convert tags to string
        st.session_state["df"][TAGS_COLUMN] = st.session_state["df"][
            TAGS_COLUMN
        ].astype(str)

        # Get zeroshot_topic_list from tags
        zeroshot_topic_list = (
            st.session_state["df"][TAGS_COLUMN]
            .fillna("")
            .str.findall(r"#\w+")
            .explode()
            .unique()
        )

        # Remove # and _ from tags and convert to string
        zeroshot_topic_list = [
            str(tag).replace("#", "").replace("_", " ")
            for tag in zeroshot_topic_list
            if tag
        ]
    # Else, set zeroshot_topic_list to None
    else:
        zeroshot_topic_list = None

    # Train topic model
    bertopic, topics = fit_bertopic(
        docs=texts_list,
        embeddings=embeddings,
        min_articles_per_topic=st.session_state["min_articles_per_topic"],
        zeroshot_topic_list=zeroshot_topic_list,
    )

    # Set session_state
    st.session_state["topic_model"] = bertopic
    st.session_state["topics"] = topics

    topic_info = bertopic.get_topic_info()
    topic_info["Representation"] = topic_info["Representation"].apply(
        lambda x: x[:TOP_N_WORDS]
    )

    st.session_state["topics_info"] = topic_info[
        topic_info["Topic"] != -1
    ]  # exclude -1 topic from topic list
