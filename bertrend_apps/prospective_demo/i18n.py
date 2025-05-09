"""
Internationalization (i18n) module for the prospective demo application.
Provides functionality for translating text between French and English.
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import streamlit as st
from typing import Dict, Any, Optional

# Available languages
LANGUAGES = {"fr": "Français", "en": "English"}

# Default language
DEFAULT_LANGUAGE = "fr"

# Translation dictionaries
TRANSLATIONS = {
    # App.py translations
    "app_title": {
        "fr": "BERTrend - Démo Veille & Analyse",
        "en": "BERTrend - Monitoring & Analysis Demo",
    },
    "tab_monitoring": {"fr": "Veilles", "en": "Monitoring"},
    "tab_models": {"fr": "Modèles", "en": "Models"},
    "tab_trends": {"fr": "Tendances", "en": "Trends"},
    "tab_analysis": {"fr": "Analyses", "en": "Analysis"},
    "tab_reports": {"fr": "Génération de rapports", "en": "Report Generation"},
    "data_flow_config": {
        "fr": "Configuration des flux de données",
        "en": "Data Flow Configuration",
    },
    "data_collection_status": {
        "fr": "Etat de collecte des données",
        "en": "Data Collection Status",
    },
    "model_status_by_monitoring": {
        "fr": "Statut des modèles par veille",
        "en": "Model Status by Monitoring",
    },
    # dashboard_analysis.py translations
    "detailed_analysis_by_topic": {
        "fr": "Analyse détaillée par sujet",
        "en": "Detailed Analysis by Topic",
    },
    "topic_selection": {"fr": "Sélection du sujet", "en": "Topic Selection"},
    "emerging_topic": {"fr": "Sujet émergent", "en": "Emerging Topic"},
    "strong_topic": {"fr": "Sujet fort", "en": "Strong Topic"},
    "nothing_to_display": {"fr": "Rien à afficher", "en": "Nothing to display"},
    # dashboard_signals.py translations
    "title": {"fr": "Titre", "en": "Title"},
    "todo_message": {
        "fr": "TODO <Place disponible pour d'autres infos...>En particulier, rajouter la courbe d'évolution des sujets en se limitant à ceux présentés dans les tableaux weak/strong/noise",
        "en": "TODO <Space available for other information...>In particular, add the topic evolution curve limited to those presented in the weak/strong/noise tables",
    },
    "explore_sources_by_topic": {
        "fr": "Exploration des sources par sujet",
        "en": "Explore Sources by Topic",
    },
    "signal_type": {"fr": "Type de signal", "en": "Signal Type"},
    "emerging_topics": {"fr": "Sujets émergents", "en": "Emerging Topics"},
    "strong_topics": {"fr": "Sujets forts", "en": "Strong Topics"},
    "no_data": {"fr": "Pas de données", "en": "No data"},
    "topic": {"fr": "Sujet", "en": "Topic"},
    "explore_sources": {"fr": "Exploration des sources", "en": "Explore Sources"},
    "reference_articles": {"fr": "Articles de référence", "en": "Reference Articles"},
    # Signal categories
    "weak_signals": {"fr": "Signaux faibles", "en": "Weak Signals"},
    "strong_signals": {"fr": "Signaux forts", "en": "Strong Signals"},
    "noise": {"fr": "Bruit", "en": "Noise"},
    "no_weak_signals": {
        "fr": "Aucun signal faible n'a été détecté à l'horodatage {timestamp}.",
        "en": "No weak signals were detected at timestamp {timestamp}.",
    },
    "no_strong_signals": {
        "fr": "Aucun signal fort n'a été détecté à l'horodatage {timestamp}.",
        "en": "No strong signals were detected at timestamp {timestamp}.",
    },
    "no_noise_signals": {
        "fr": "Aucun signal de bruit n'a été détecté à l'horodatage {timestamp}.",
        "en": "No noisy signals were detected at timestamp {timestamp}.",
    },
}


def get_current_language() -> str:
    """Get the currently selected language from the session state."""
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
    return st.session_state.language


def set_language(language: str) -> None:
    """Set the current language in the session state."""
    if language in LANGUAGES:
        st.session_state.language = language


def create_language_selector() -> None:
    """Create a language selector widget in the sidebar."""
    current_lang = get_current_language()
    selected_lang = st.sidebar.selectbox(
        "Language / Langue",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=list(LANGUAGES.keys()).index(current_lang),
    )

    if selected_lang != current_lang:
        set_language(selected_lang)
        st.rerun()


def translate(key: str, default: Optional[str] = None) -> str:
    """
    Translate a text key to the current language.

    Args:
        key: The translation key to look up
        default: Default text to return if the key is not found

    Returns:
        The translated text in the current language
    """
    lang = get_current_language()

    if key in TRANSLATIONS and lang in TRANSLATIONS[key]:
        return TRANSLATIONS[key][lang]

    # If the key doesn't exist or the language is not available for this key
    if default:
        return default
    return key  # Return the key itself as fallback
