"""
Internationalization (i18n) module for the prospective demo application.
Provides functionality for translating text between French and English.
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from typing import Optional

import streamlit as st

from bertrend.demos.topic_analysis.i18n_translations import (
    TRANSLATIONS as TOPIC_ANALYSIS_TRANSLATIONS,
)
from bertrend.demos.demos_utils.i18n_translations import (
    TRANSLATIONS as DEMOS_UTILS_TRANSLATIONS,
)
from bertrend.demos.weak_signals.i18n_translations import (
    TRANSLATIONS as WEAK_SIGNALS_TRANSLATIONS,
)

# Merge all translations
TRANSLATIONS = {
    **DEMOS_UTILS_TRANSLATIONS,
    **TOPIC_ANALYSIS_TRANSLATIONS,
    **WEAK_SIGNALS_TRANSLATIONS,
}

# Available languages
LANGUAGES = {"fr": "Français", "en": "English"}

# Default language
DEFAULT_LANGUAGE = "en"


def get_current_internationalization_language() -> str:
    """Get the currently selected language from the session state."""
    if "internationalization_language" not in st.session_state:
        st.session_state.internationalization_language = DEFAULT_LANGUAGE
    return st.session_state.internationalization_language


def set_internationalization_language(language: str) -> None:
    """Set the current language in the session state."""
    if language in LANGUAGES:
        st.session_state.internationalization_language = language


def create_internationalization_language_selector() -> None:
    """Create a language selector widget in the sidebar."""
    current_lang = get_current_internationalization_language()
    selected_lang = st.sidebar.selectbox(
        "Language / Langue",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=list(LANGUAGES.keys()).index(current_lang),
    )

    if selected_lang != current_lang:
        set_internationalization_language(selected_lang)
        st.rerun()


def translate_helper(key: str, default: Optional[str] = None, translations=None) -> str:
    """
    Translate a text key to the current language.

    Args:
        key: The translation key to look up
        default: Default text to return if the key is not found

    Returns:
        The translated text in the current language
    """
    if translations is None:
        return key
    lang = get_current_internationalization_language()

    if key in translations and lang in translations[key]:
        return translations[key][lang]

    # If the key doesn't exist, or the language is not available for this key
    if default:
        return default
    return key  # Return the key itself as fallback


def translate(key: str, default: Optional[str] = None) -> str:
    """
    Translate a text key to the current language.

    Args:
        key: The translation key to look up
        default: Default text to return if the key is not found

    Returns:
        The translated text in the current language
    """
    return translate_helper(key, default, TRANSLATIONS)
