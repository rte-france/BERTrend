#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import streamlit as st

from bertrend.bertrend_apps.exploration.curebot.tabs import tab1, tab2, tab3

# Set wide layout
st.set_page_config(
    page_title="Curebot - Exploration de sujets",
    layout="wide",
)

# Set app title
st.title("Curebot - Exploration de sujets")

# Set sidebar
with st.sidebar:
    with st.expander("Paramètres"):
        st.checkbox(
            "Utiliser les tags",
            key="use_tags",
            value=False,
            help="Utiliser les tags Curebot pour orienter la recherche de sujets.",
        )
        st.number_input(
            "Nombre d'articles minimum par sujet",
            2,
            50,
            5,
            key="min_articles_per_topic",
            help="Permet d'influencer le nombre total de sujets trouvés par le modèle. Plus ce nombre est élevé, moins il y aura de sujets.",
        )

# Create tabs
tab1_content, tab2_content, tab3_content = st.tabs(
    ["Données", "Résultats", "Newsletter"]
)

# Main tabs
with tab1_content:
    tab1.show()

with tab2_content:
    tab2.show()

with tab3_content:
    tab3.show()
