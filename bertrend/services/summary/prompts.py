#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

###################### SUMMARY PROMPTS

FR_SYSTEM_SUMMARY_WORDS = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    "Générez un résumé en {num_words} mots maximum de ce texte."
    "Essayez de retenir les points les plus importants, "
    "en fournissant un résumé cohérent et lisible qui pourrait aider une personne à comprendre "
    "les points principaux du texte sans avoir besoin de lire le texte en entier. "
    "Eviter les détails inutiles."
)
# num_words: number of words the summary should contain

EN_SYSTEM_SUMMARY_WORDS = (
    "You are a highly qualified AI, trained in language understanding and synthesis. "
    "Generate a summary of the following text in a maximum of {num_words} words."
    "Try to capture the most important points, "
    "providing a coherent and readable summary that could help a person understand "
    "the main points of the text without needing to read the entire text. "
    "Please avoid unnecessary details."
)
# num_words: number of words the summary should contain


FR_USER_SUMMARY_WORDS = FR_SYSTEM_SUMMARY_WORDS + (" Texte :\n {text}")
# num_words: number of words the summary should contain
# text: text to be summarized

EN_USER_SUMMARY_WORDS = EN_SYSTEM_SUMMARY_WORDS + (" Text :\n {text}")
# num_words: number of words the summary should contain
# text: text to be summarized

FR_SYSTEM_SUMMARY_SENTENCES = FR_SYSTEM_SUMMARY_WORDS.replace(
    "{num_words} mots", "{num_sentences} phrases"
)
EN_SYSTEM_SUMMARY_SENTENCES = EN_SYSTEM_SUMMARY_WORDS.replace(
    "{num_words} words", "{num_sentences} sentences"
)
# num_sentences: number of sentences the summary should contain
