#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

FR_USER_SUMMARY_MULTIPLE_DOCS = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    "Voici ci-dessous plusieurs articles de presse (Titre et Contenu). "
    "Tous les articles appartiennent au même thème représenté par les mots-clés suivants : {keywords}. "
    "Générez une synthèse en {nb_sentences} phrases maximum de ces articles qui doit être en lien avec le thème évoqué par les mots-clés. "
    "La synthèse doit permettre de donner une vision d'ensemble du thème sans lire les articles. "
    "Ne pas commencer par 'Les articles...' mais commencer directement la synthèse.\n"
    "Liste des articles :\n"
    "```{article_list}```\n"
    "Synthèse :"
)

EN_USER_SUMMARY_MULTIPLE_DOCS = (
    "You are a highly qualified AI, trained in language understanding and synthesis. "
    "Below are several press articles (Title and Content). "
    "All the articles belong to the same topic represented by the following keywords: {keywords}. "
    "Generate a summary of these articles, which must be related to the theme evoked by the keywords. "
    "The summary must not exceed {nb_sentences} sentences and should include the essential information from the articles.\n"
    "List of articles :\n"
    "```{article_list}```\n"
    "Summary :"
)
# keywords: list of keywords describing the topic
# list of articles and their title


###################### TOPIC PROMPTS

FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES = (
    "Décrit en une courte expression le thème associé à l'ensemble des extraits "
    "suivants. Le thème doit être court et spécifique en 4 mots maximum. "
    '\n"{title_list}"'
)

FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES_V2 = (
    'Dans le cadre de la génération d\'une newsletter sur le thème "{newsletter_title}", '
    "décrit en une courte expression le sous-thème associé au texte suivant. "
    "L'expression doit être courte en 4 mots maximum. "
    "Elle doit montrer en quoi le texte est spécifique pour le thème. "
    "Elle ne doit pas être générale ni décrire un sous-thème trop proche du thème de la newsletter.\n\n"
    'Texte : "{title_list}"'
)

# title_list: list of documents extracts belonging to the topic

EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES_V2 = (
    "Describe in a short sentence the topic associated with the following extracts. "
    "The topic description should be short and specific, no more than 4 words. "
    '\n"{title_list}"'
)

EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES_V2 = (
    'Within the framework of the generation of a newsletter on the topic "{newsletter_title}", '
    "describe in a short sentence the sub-topic associated with the following text. "
    "The sentence should be short, no more than 4 words. "
    "It should show in what way the text is specific to the topic. "
    "It should not be general nor describe a sub-topic too close to the topic of the newsletter.\n\n"
    '\n"{title_list}"'
)
# title_list: list of documents extracts belonging to the topic

FR_USER_GENERATE_TOPIC_LABEL_TITLE = (
    "Vous êtes une IA hautement qualifiée, formée à la compréhension et à la synthèse du langage. "
    'Après utilisation d\'un algorithme de topic modelling, un topic est représenté par les mots-clé suivants : """{keywords}.""" '
    'Le topic contient plusieurs documents dont les titres sont les suivants :\n"""\n{title_list}\n"""\n'
    "À partir de ces informations sur le topic, écrivez un titre court de ce topic en 3 mots maximum. "
)
# keywords: list of keywords describing the topic
# title_list: list of documents title belonging to the topic


FRENCH_TOPIC_REPRESENTATION_PROMPT = (
    "J'ai un topic qui contient les documents suivants :\n"
    "[DOCUMENTS]\n"
    "Le topic est décrit par les mots-clés suivants : [KEYWORDS]\n"
    "Sur la base des informations ci-dessus, extraire une courte étiquette de topic dans le format suivant :\n"
    "Topic : <étiquette du sujet>"
)
# Passed directly to BERTopic's OpenAI wrapper, formatted similar to BERTopic's original prompt which can be found in its source code
