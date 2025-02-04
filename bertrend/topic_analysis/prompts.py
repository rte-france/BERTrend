#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

TOPIC_DESCRIPTION_PROMPT_FR = """En tant qu'expert en analyse thématique, votre tâche est de générer un titre et une description pour un thème spécifique.


    Représentation du thème : {topic_representation}

    Voici le contexte de ce thème :

    {docs_text}

    Basé sur ces informations, veuillez fournir :
    1. Un titre concis et informatif pour ce thème (maximum 10 mots)
    2. Une description détaillée du thème (environ 100 mots)

    Réponse au format JSON:
    title: [Votre titre ici], description: [Votre description ici]
    """

TOPIC_DESCRIPTION_PROMPT_EN = """As a topic analysis expert, your task is to generate a title and a description for a specific theme.

Representation of the topic: {topic_representation}

Here is the context of this topic:

{docs_text}

Based on this information, please provide:
1. A concise and informative title for this theme (maximum 10 words)
2. A detailed description of the theme (about 100 words)

Response in JSON format :
title: [Your title here], description: [Your description here]
"""

TOPIC_DESCRIPTION_PROMPT = {
    "en": TOPIC_DESCRIPTION_PROMPT_EN,
    "fr": TOPIC_DESCRIPTION_PROMPT_FR,
}
