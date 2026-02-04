#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger

from bertrend import OUTPUT_PATH
from bertrend.trend_analysis.data_structure import SignalAnalysis, TopicSummaryList

# Global variables for prompts
SIGNAL_INTRO = {
    "en": """As an elite strategic foresight analyst with extensive expertise across multiple domains and industries, your task is to conduct a comprehensive evaluation of a potential signal derived from the following topic summary:

{summary_from_first_prompt}

Leverage your knowledge and analytical skills to provide an in-depth analysis of this signal's potential impact and evolution:
""",
    "fr": """En tant qu'analyste de prospective stratégique d'élite avec une expertise étendue dans de multiples domaines et industries, votre tâche est de mener une évaluation complète d'un signal potentiel dérivé du résumé de sujet suivant :

{summary_from_first_prompt}

Utilisez vos connaissances et compétences analytiques pour fournir une analyse approfondie de l'impact potentiel et de l'évolution de ce signal :
""",
}

SIGNAL_INSTRUCTIONS = {
    "en": """
Analyze this signal only if the input data is sufficiently complete and substantive. If the subject summary lacks completeness, substance, specificity, or novelty, respond with an empty JSON dictionary: {}

=== DECISION RULES (STRICT SUFFICIENCY) ===
- Overall Sufficiency: Proceed with analysis only if the topic summary contains enough concrete, non-trivial information to support evidence-based reasoning across at least one analysis dimension (Impact, Evolution, Interconnections, Drivers/Inhibitors).
- Section Sufficiency: For each analysis section (Impact, Evolution Scenarios, Interconnections and Synergies, Drivers and Inhibitors), include the section ONLY if the input contains enough detail to meet the Minimum Quality Requirements. If not, omit the section entirely (do not include placeholders).
- Evidence-First: Prefer omission over speculation. Do not infer facts not supported by the provided summary.

For substantial signals, provide the following sections (include only those that meet the standards; omit any that do not):

1. Potential Impact Analysis:
   - Examine potential effects on sectors, industries, and societal aspects.
   - Cover immediate (1–2 years), medium (3–5 years), and long-term (5–10 years) horizons.
   - Include ripple effects and second-order consequences when supportable.

2. Evolution Scenarios:
   - Describe plausible future developments and manifestation paths.
   - Identify factors shaping the trajectory.
   - Provide both optimistic and pessimistic scenarios only if both are sufficiently supported; otherwise include the supported one(s) and omit the rest.

3. Interconnections and Synergies:
   - Identify interactions with current trends or emerging phenomena.
   - Discuss synergies or conflicts with existing systems or paradigms.

4. Drivers and Inhibitors:
   - Analyze accelerants and amplifiers of the signal.
   - Examine barriers, constraints, or resistances.

Your analysis should be thorough and nuanced, going beyond surface-level observations, and grounded in the provided content. Make well-reasoned predictions only when they are logically supported by the input. If analysis cannot be substantiated with clear reasoning, omit that section.

=== OUTPUT QUALITY STANDARDS ===
Avoid:
- Vague generalizations
- Obvious conclusions without new insight
- Insufficient evidence
- Generic observations
- Circular reasoning
- Superficial treatment
- Unsubstantiated speculation
- Outdated perspectives

=== MINIMUM QUALITY REQUIREMENTS (APPLY PER SECTION) ===
Each included section must demonstrate at least 2 of:
- Specific Context: Clear temporal, geographic, or sectoral boundaries with examples
- Concrete Evidence: Quantifiable insights, verifiable examples, or substantiated claims from the input
- Novel Perspectives: Non-obvious connections or emerging patterns grounded in the input
- Actionable Intelligence: Guidance that enables decision-making or planning
- Cross-Domain Impact: Implications across multiple sectors or domains
- Measurable Dimensions: Metrics, indicators, or tracking mechanisms
- Causal Analysis: Clear cause-and-effect or contributing factors
- Strategic Relevance: Direct link to business, policy, or societal decisions

=== OUTPUT REQUIREMENTS ===
- Section-Level Omission: Omit any section that cannot meet the Minimum Quality Requirements with the provided information.
- Evidence-Based: Use specific, quantifiable language with concrete examples derived from the input.
- Confidence Levels: Clearly distinguish high-confidence assessments from speculative insights.
- Decision-Focused: Prioritize actionable intelligence for strategic decision-makers.
- Balanced Objectivity: Maintain rigor while acknowledging uncertainties and limitations.
- Temporal Structure: Organize insights across immediate (1–2 years), medium (3–5 years), and long-term (5–10 years) where applicable.
- Final Validation: Before finalizing, remove any statement that cannot be traced to or logically derived from the provided summary.

If no section can meet these standards, return an empty JSON dictionary: {}
""",
    "fr": """
Analysez ce signal uniquement si les données d’entrée sont suffisamment complètes et substantielles. Si le résumé du sujet manque de complétude, de substance, de spécificité ou de nouveauté, répondez avec un dictionnaire JSON vide : {}

=== RÈGLES DE DÉCISION (SUFFISANCE STRICTE) ===
- Suffisance Globale : Poursuivez l’analyse uniquement si le résumé contient des informations concrètes et non triviales permettant un raisonnement fondé sur des preuves pour au moins une dimension d’analyse (Impact, Évolution, Interconnexions, Moteurs/Inhibiteurs).
- Suffisance par Section : Pour chaque section (Impact, Scénarios d’Évolution, Interconnexions et Synergies, Moteurs et Inhibiteurs), incluez la section UNIQUEMENT si les informations permettent d’atteindre les Exigences Minimales de Qualité. Sinon, omettez entièrement la section (pas d’espace réservé).
- Primauté des Preuves : Préférez l’omission à la spéculation. N’inférez pas de faits non étayés par le résumé fourni.

Pour les signaux substantiels, fournissez les sections suivantes (n’incluez que celles qui respectent les standards ; omettez celles qui ne les atteignent pas) :

1. Analyse de l’Impact Potentiel :
   - Effets potentiels sur secteurs, industries et aspects sociétaux.
   - Couvrez les horizons immédiat (1–2 ans), moyen (3–5 ans) et long terme (5–10 ans).
   - Incluez les effets d’entraînement et de second ordre lorsque c’est étayé.

2. Scénarios d’Évolution :
   - Décrivez des développements plausibles et voies de manifestation futures.
   - Identifiez les facteurs influençant la trajectoire.
   - Fournissez des scénarios optimistes et pessimistes uniquement si tous deux sont suffisamment étayés ; sinon, incluez seulement ceux qui le sont et omettez le reste.

3. Interconnexions et Synergies :
   - Interactions avec les tendances actuelles ou phénomènes émergents.
   - Synergies ou conflits avec les systèmes ou paradigmes existants.

4. Moteurs et Inhibiteurs :
   - Facteurs accélérateurs/amplificateurs du signal.
   - Obstacles, contraintes ou résistances.

Votre analyse doit être approfondie, nuancée et ancrée dans le contenu fourni. Proposez des prédictions bien raisonnées uniquement lorsqu’elles sont logiquement étayées par l’entrée. Si une section ne peut être étayée par un raisonnement clair, omettez-la.

=== STANDARDS DE QUALITÉ DE SORTIE ===
À éviter :
- Généralisations vagues
- Conclusions évidentes sans nouveaux éclairages
- Preuves insuffisantes
- Observations génériques
- Raisonnement circulaire
- Traitement superficiel
- Spéculation non étayée
- Perspectives obsolètes

=== EXIGENCES MINIMALES DE QUALITÉ (PAR SECTION) ===
Chaque section incluse doit démontrer au moins 2 des éléments suivants :
- Contexte Spécifique : Limites temporelles, géographiques ou sectorielles claires avec exemples
- Preuves Concrètes : Insights quantifiables, exemples vérifiables, ou affirmations étayées par l’entrée
- Perspectives Nouvelles : Connexions non évidentes ou schémas émergents ancrés dans l’entrée
- Intelligence Actionnable : Conseils permettant la décision ou la planification
- Impact Trans-Domaine : Implications multi-sectorielles
- Dimensions Mesurables : Métriques, indicateurs ou mécanismes de suivi
- Analyse Causale : Relations de cause à effet ou facteurs contributifs clairs
- Pertinence Stratégique : Lien direct avec décisions business, politiques ou sociétales

=== EXIGENCES DE SORTIE ===
- Omission par Section : Omettez toute section qui ne peut atteindre les exigences minimales avec les informations disponibles.
- Basé sur les Preuves : Utilisez un langage spécifique et quantifiable avec des exemples concrets tirés de l’entrée.
- Niveaux de Confiance : Distinguez clairement les évaluations à haute confiance des insights spéculatifs.
- Orienté Décision : Priorisez l’intelligence actionnable pour les décideurs stratégiques.
- Objectivité Équilibrée : Rigueur analytique en reconnaissant les incertitudes et limitations.
- Structure Temporelle : Organisez les insights sur les horizons immédiat (1–2 ans), moyen (3–5 ans) et long terme (5–10 ans) lorsque pertinent.
- Validation Finale : Avant de finaliser, retirez toute affirmation qui ne peut être retracée ou logiquement dérivée du résumé fourni.

Si aucune section ne peut respecter ces standards, renvoyez un dictionnaire JSON vide : {}
""",
}

TOPIC_SUMMARY_PROMPT = {
    "en": """
As an expert analyst specializing in trend analysis and strategic foresight, your task is to provide a comprehensive evolution summary of Topic {topic_number}. Use only the information provided below:

{content_summary}

Structure your analysis as follows:

For the first timestamp:

## [Concise yet impactful title capturing the essence of the topic at this point]
### Date: [Relevant date or time frame - format %Y-%m-%d]
### Key Developments
- [Bullet point summarizing a major development or trend]
- [Additional bullet points as needed]

### Analysis
[2-3 sentences maximum providing deeper insights into the developments, their potential implications, and their significance in the broader context of the topic's evolution]

For all subsequent timestamps:

## [Concise yet impactful title capturing the essence of the topic at this point]
### Date: [Relevant date or time frame - format %Y-%m-%d]
### Key Developments
- [Bullet point summarizing a major development or trend]
- [Additional bullet points as needed]

### Analysis
[2-3 sentences maximum providing deeper insights into the developments, their potential implications, and their significance in the broader context of the topic's evolution]

### What's New
[1-2 sentences maximum highlighting how this period differs from the previous one, focusing on new elements or significant changes]

Provide your analysis using only this format, based solely on the information given. Do not include any additional summary or overview sections beyond what is specified in this structure.
""",
    "fr": """
En tant qu'analyste expert spécialisé dans l'analyse des tendances et la prospective stratégique, votre tâche est de fournir un résumé complet de l'évolution du Sujet {topic_number}. Utilisez uniquement les informations fournies ci-dessous :

{content_summary}

Structurez votre analyse comme suit :

Pour le premier timestamp :

## [Titre concis mais percutant capturant l'essence du sujet à ce moment]
### Date : [Date ou période pertinente - format %Y-%m-%d]
### Développements Clés
- [Point résumant un développement majeur ou une tendance]
- [Points supplémentaires si nécessaire]

### Analyse
[2-3 phrases maximum fournissant des insights plus profonds sur les développements, leurs implications potentielles et leur importance dans le contexte plus large de l'évolution du sujet]

Pour tous les timestamps suivants :

## [Titre concis mais percutant capturant l'essence du sujet à ce moment]
### Date : [Date ou période pertinente - format %Y-%m-%d]
### Développements Clés
- [Point résumant un développement majeur ou une tendance]
- [Points supplémentaires si nécessaire]

### Analyse
[2-3 phrases maximum fournissant des insights plus profonds sur les développements, leurs implications potentielles et leur importance dans le contexte plus large de l'évolution du sujet]

### Nouveautés
[1-2 phrases maximum soulignant en quoi cette période diffère de la précédente, en se concentrant sur les nouveaux éléments ou les changements significatifs]

Fournissez votre analyse en utilisant uniquement ce format, basé uniquement sur les informations données. N'incluez pas de sections de résumé ou d'aperçu supplémentaires au-delà de ce qui est spécifié dans cette structure.
""",
}


def get_prompt(
    language: str,
    prompt_type: str,
    topic_number: int = None,
    content_summary: str = None,
    summary_from_first_prompt: str = None,
):
    lang = "en" if language == "English" else "fr"

    if prompt_type == "weak_signal":
        prompt = (
            SIGNAL_INTRO[lang].format(
                summary_from_first_prompt=summary_from_first_prompt
            )
            + SIGNAL_INSTRUCTIONS[lang]
        )

    elif prompt_type == "topic_summary":
        prompt = TOPIC_SUMMARY_PROMPT[lang].format(
            topic_number=topic_number, content_summary=content_summary
        )
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    return prompt


def save_html_output(html_output, output_file="signal_llm.html"):
    """Function to save the model's output as HTML"""
    output_path = OUTPUT_PATH / output_file

    # Save the cleaned HTML
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_output)
    logger.debug(f"Cleaned HTML output saved to {output_path}")


def fill_html_template(
    topic_summary_list: TopicSummaryList,
    signal_analysis: SignalAnalysis,
    language: str = "fr",
) -> str:
    """Fill the HTML template with appropriate data"""
    # Setup Jinja2 environment
    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(
        loader=FileSystemLoader(template_dir),
    )
    template = env.get_template(
        "signal_llm_template_en.html"
        if language == "en"
        else "signal_llm_template_fr.html"
    )

    # Sort the list by date from most recent to least recent
    try:
        sorted_topic_summary_by_time_period = sorted(
            topic_summary_list.topic_summary_by_time_period,
            key=lambda x: datetime.strptime(x.date, "%Y-%m-%d"),
            reverse=True,
        )
        topic_summary_list.topic_summary_by_time_period = (
            sorted_topic_summary_by_time_period
        )
    except Exception as e:
        logger.warning("Cannot sort summaries by date, probably wrong date format")

    # Render the template with the provided data
    rendered_html = template.render(
        topic_summary_list=topic_summary_list, signal_analysis=signal_analysis
    )

    # FIXME: many \n are added...
    rendered_html = rendered_html.replace("\n", "")
    rendered_html = rendered_html.replace("\\'", "'")

    return rendered_html
