"""
Internationalization (i18n) module for the prospective demo application.
Provides functionality for translating text between French and English.
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from typing import Optional

from bertrend.demos.demos_utils.i18n import translate_helper
from bertrend_apps.prospective_demo.i18n_translations import (
    TRANSLATIONS as PROSPECTIVE_TRANSLATIONS,
)
from bertrend.demos.demos_utils.i18n_translations import (
    TRANSLATIONS as DEMOS_UTILS_TRANSLATIONS,
)


# Merge all translations
DEMO_TRANSLATIONS = {
    **DEMOS_UTILS_TRANSLATIONS,
    **PROSPECTIVE_TRANSLATIONS,
}


def translate(key: str, default: Optional[str] = None) -> str:
    """
    Translate a text key to the current language.

    Args:
        key: The translation key to look up
        default: Default text to return if the key is not found

    Returns:
        The translated text in the current language
    """
    return translate_helper(key, default, DEMO_TRANSLATIONS)
