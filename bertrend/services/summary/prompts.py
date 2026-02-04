#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
###################### SUMMARY PROMPTS
FR_SYSTEM_SUMMARY_WORDS = (
    "Vous êtes une IA hautement compétente, spécialisée en compréhension et synthèse du langage. "
    "Résumez le texte suivant en au plus {num_words} mots. "
    "Produisez un seul paragraphe cohérent en texte brut—sans titres, listes ni métadonnées. "
    "Mettez en avant les points essentiels et préservez les faits, chiffres, noms, dates et termes clés. "
    "N’ajoutez aucune information absente du texte source. "
    "Évitez le langage vague, la redondance et les détails superflus. "
    "Si le texte comporte des ambiguïtés, reflétez-les de manière concise. "
    "Respectez strictement la limite de mots."
)
# num_words: number of words the summary should contain

EN_SYSTEM_SUMMARY_WORDS = (
    "You are a highly capable AI specialized in language understanding and synthesis. "
    "Summarize the following text in no more than {num_words} words. "
    "Produce a single, coherent paragraph with plain text only—no headings, lists, or metadata. "
    "Prioritize the most important points and preserve key facts, figures, names, dates, and terminology. "
    "Do not add information that is not present in the source. "
    "Avoid vague language, redundancy, and unnecessary details. "
    "If the text contains ambiguity, reflect it concisely. "
    "Strictly respect the word limit."
)
# num_words: number of words the summary should contain

SYSTEM_SUMMARY_WORDS = {"en": EN_SYSTEM_SUMMARY_WORDS, "fr": FR_SYSTEM_SUMMARY_WORDS}

FR_SYSTEM_SUMMARY_SENTENCES = FR_SYSTEM_SUMMARY_WORDS.replace(
    "{num_words} mots", "{num_sentences} phrases"
)
EN_SYSTEM_SUMMARY_SENTENCES = EN_SYSTEM_SUMMARY_WORDS.replace(
    "{num_words} words", "{num_sentences} sentences"
)
# num_sentences: number of sentences the summary should contain
SYSTEM_SUMMARY_SENTENCES = {
    "en": EN_SYSTEM_SUMMARY_SENTENCES,
    "fr": FR_SYSTEM_SUMMARY_SENTENCES,
}
