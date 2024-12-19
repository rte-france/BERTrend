#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os

import streamlit as st
from loguru import logger

from bertrend.services.summary.abstractive_summarizer import AbstractiveSummarizer
from bertrend.services.summary.chatgpt_summarizer import GPTSummarizer
from bertrend.services.summary.extractive_summarizer import (
    EnhancedExtractiveSummarizer,
    ExtractiveSummarizer,
)

SUMMARIZER_OPTIONS_MAPPER = {
    "GPTSummarizer": GPTSummarizer,
    "EnhancedExtractiveSummarizer": EnhancedExtractiveSummarizer,
    "ExtractiveSummarizer": ExtractiveSummarizer,
    "AbstractiveSummarizer": AbstractiveSummarizer,
}

DEFAULT_TEXT = """RTE, sigle du Réseau de transport d'électricité, est le gestionnaire de réseau de transport français responsable du réseau public de transport d'électricité haute tension en France métropolitaine (la Corse n'est pas gérée par RTE). Sa mission fondamentale est d’assurer à tous ses clients l’accès à une alimentation électrique économique, sûre et propre. RTE connecte ses clients par une infrastructure adaptée et leur fournit tous les outils et services qui leur permettent d’en tirer parti pour répondre à leurs besoins. À cet effet, RTE exploite, maintient et développe le réseau à haute et très haute tension. Il est le garant du bon fonctionnement et de la sûreté du système électrique. RTE achemine l’électricité entre les fournisseurs d’électricité (français et européens) et les consommateurs, qu’ils soient distributeurs d’électricité ou industriels directement raccordés au réseau de transport. Plus de 105 000 km de lignes comprises entre 45 000 et 400 000 volts et 50 lignes transfrontalières connectent le réseau français à 33 pays européens, offrant ainsi des opportunités d’échanges d’électricité essentiels pour l’optimisation économique du système électrique.

RTE fait partie du Réseau européen des gestionnaires de réseau de transport d’électricité (ENTSO-E), organisation qui regroupe les gestionnaires de réseaux de transport à haute et très haute tension de 35 pays3. Elle découle de la fusion le 1er juillet 2009 de l'UCTE, de BALTSO, de NORDEL, d'ATSOI et d'UKTSOA.

Les réseaux de ces sociétés desservent, via les réseaux de distribution, une population d'environ 500 millions de personnes. Ils se décomposent en cinq grands systèmes synchrones : l'Europe continentale (à laquelle est rattachée la Turquie), les pays baltes, les pays nordiques, l'Irlande et la Grande-Bretagne.

Les lignes à basse et moyenne tension françaises ne sont pas du ressort de RTE. Elles sont essentiellement exploitées par Enedis (anciennement ERDF, filiale de distribution électrique d'EDF), mais aussi par d'autres entreprises locales de distribution (ELD) comme Électricité de Strasbourg, l'Usine d'électricité de Metz, ou encore Gascogne Énergies Services à Aire-sur-l’Adour dans les Landes."""

models = {}


@st.cache_resource
def get_summarizer(summary_model):
    """Instantiates models once, keep them in memory"""
    kwargs = {}
    if summary_model == "GPTSummarizer":
        kwargs["api_key"] = st.session_state.openai_api_key
        kwargs["endpoint"] = st.session_state.openai_endpoint

    summarizer_class = SUMMARIZER_OPTIONS_MAPPER[summary_model]
    return summarizer_class(**kwargs)


def app():
    st.title("Comparison of summarizers")

    st.warning(
        "Warning: be sure that when using a GPT summarizer, the server is on Azure or local. "
        "Otherwise, use it EXCLUSIVELY on public data."
    )

    summary_model = st.selectbox(
        "summary model", SUMMARIZER_OPTIONS_MAPPER.keys(), key="summary_model"
    )
    summary_ratio = (
        st.number_input(
            "summary ratio", min_value=1, max_value=100, value=20, key="summary_ratio"
        )
        / 100
    )
    st.text_input(
        "Openai API key",
        key="openai_api_key",
        value=os.getenv("OPENAI_API_KEY"),
        type="password",
    )
    st.text_input(
        "Openai endpoint", key="openai_endpoint", value=os.getenv("OPENAI_ENDPOINT")
    )
    st.text_input(
        "Openai model name",
        key="openai_model_name",
        value=os.getenv("OPENAI_DEFAULT_MODEL_NAME"),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Input text", DEFAULT_TEXT, height=250, key="text")
    with col2:
        st.text_area("Summary", "", height=250, key="summary")

    def on_click():
        summarizer = get_summarizer(summary_model)
        st.session_state.summary = summarizer.generate_summary(
            st.session_state.text, max_length_ratio=summary_ratio
        )
        logger.debug(st.session_state.summary)

    st.button("Summarize", on_click=on_click)


if __name__ == "__main__":
    app()
