import locale
from pathlib import Path

import jinja2
import streamlit as st

from bertrend.llm_utils.newsletter_features import render_newsletter_html
from bertrend_apps.exploration.curebot.app_utils import (
    NEWSLETTER_TEMPLATE,
    create_newsletter,
)

# Set french locale
locale.setlocale(locale.LC_ALL, "fr_FR.UTF-8")

CSS_STYLE = (
    Path(__file__).parent.parent.parent.parent.parent
    / "bertrend/llm_utils/newsletter.css"
)


def show() -> None:
    if "topic_model" not in st.session_state:
        st.warning("Veuillez ajouter des données et lancer la détection des sujets.")
    else:
        # Show newsletter parameters
        newsletter_parameters()

        # Columns for buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        # Show newsletter creation button
        with col1:
            if st.button("Créer la newsletter"):
                with st.spinner("Création de la newsletter..."):
                    get_newsletter()

        # Show newsletter edition button
        with col2:
            st.button(
                "Éditer",
                on_click=edit_newsletter,
                disabled=not "newsletter_text" in st.session_state,
            )

        # Show newsletter download button
        with col3:
            # If newsletter created, set it as data to download
            if "newsletter_text" in st.session_state:
                data = render_newsletter_html(
                    st.session_state["newsletter_text"], css_style=CSS_STYLE
                )
            # Else, set data to empty string
            else:
                data = ""

            # Download button
            st.download_button(
                "Télécharger",
                file_name="newsletters.html",
                mime="text/html",
                data=data,
                disabled=not "newsletter_text" in st.session_state,
            )

        # Show newsletter
        if "newsletter_text" in st.session_state:
            st.write(st.session_state["newsletter_text"], unsafe_allow_html=True)


def newsletter_parameters() -> None:
    """
    Show newsletter parameters:
    - Number of topics to include in the newsletter
    - Number of articles per topic to include in the newsletter
    """
    col31, col32 = st.columns([1, 1])
    with col31:
        # Select number of topics to include in the newsletter
        st.slider(
            "Nombre de sujets",
            1,
            len(st.session_state["topics_info"]),
            value=min(len(st.session_state["topics"]), 3),
            key="newsletter_nb_topics",
        )
    with col32:
        # Select number of articles per topic to include in the newsletter
        st.slider(
            "Nombre d'articles par sujet",
            1,
            10,
            value=4,
            key="newsletter_nb_articles_per_topic",
        )


def get_newsletter() -> None:
    """
    Create a newsletter based on the selected topics and articles.
    Sets in sessions_state:
    - newsletter_dict: a dictionary containing the newsletter data
    - newsletter_text: the newsletter text
    """
    # Newsletter dict
    st.session_state["newsletter_dict"] = create_newsletter(
        st.session_state["df"],
        st.session_state["topics_info"],
        st.session_state["newsletter_nb_topics"],
        st.session_state["newsletter_nb_articles_per_topic"],
    )

    # Newsletter text
    template = jinja2.Template(NEWSLETTER_TEMPLATE)
    st.session_state["newsletter_text"] = template.render(
        st.session_state["newsletter_dict"]
    )


@st.dialog("Éditer la newsletter", width="large")
def edit_newsletter() -> None:
    edited_newsltter = st.text_area(
        "",
        value=st.session_state["newsletter_text"],
        height=500,
    )
    st.session_state["newsletter_text"] = edited_newsltter
    if st.button("Enregistrer", type="primary"):
        st.rerun()
