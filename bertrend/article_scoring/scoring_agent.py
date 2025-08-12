import asyncio

from bertrend.article_scoring.article_scoring import ArticleScore
from bertrend.article_scoring.prompts import ARTICLE_SCORING_PROMPT
from bertrend.llm_utils.agent_utils import (
    BaseAgentFactory,
    AsyncAgentConcurrentProcessor,
    progress_reporter,
)


async def score_articles(articles: list[str]):
    agent = BaseAgentFactory().create_agent(
        name="scoring_agent",
        instructions=ARTICLE_SCORING_PROMPT,
        output_type=ArticleScore,
    )

    # Initialize processor
    processor = AsyncAgentConcurrentProcessor(agent=agent, max_concurrent=10)

    # Uncomment to process multiple items:
    results = await processor.process_list_concurrent(
        articles, progress_callback=progress_reporter, chunk_size=10
    )
    return results


if __name__ == "__main__":
    article = """
    Vigilance rouge canicule : quatorze départements en alerte ce mardi, la vigilance maintenue demain

L’épisode de forte chaleur se poursuit sur la majeure partie de l’Hexagone.
Par Le HuffPost avec AFP
La canicule se poursuit avec 14 département en vigilance rouge ce mardi et demain (photo d’illustration à Toulouse)
Alain Pitton / NurPhoto / Getty Images
La canicule se poursuit avec 14 département en vigilance rouge ce mardi et demain (photo d’illustration à Toulouse)

METEO - Pas de changement de couleur. La vigilance rouge en vigueur ce mardi 12 août dans 14 départements du Sud-Ouest et du Centre-Est en proie à la canicule sera maintenue mercredi13 août, a annoncé Météo-France dans son dernier bulletin publié à 6h. Le prévisionniste maintient également en vigilance orange les 64 départements sous ce niveau d’alerte.

Après des températures déjà étouffantes lundi, les départements de la Gironde, de la Dordogne, de Lot-et-Garonne, des Landes, du Lot, de Tarn-et-Garonne, du Gers, du Tarn, de la Haute-Garonne et de l’Aude restent en vigilance rouge ce mardi, rejoints à 12h par le Rhône, l’Isère, la Drôme, et l’Ardèche. Une liste valable également pour mercredi 13 août, donc.
Lire aussi
Vigilance canicule : d’où vient la vague de chaleur qui va faire exploser les températures partout cette semaine ?

« Ce mardi, les températures restent très élevées sur le Sud-Ouest, à peine inférieures à celle de lundi », explique Météo-France. « Les 40°C sont même possibles sur le littoral aquitain par endroits. Toujours des pointes à 42°C dans l’Aude. » Dans la vallée du Rhône et dans le Lyonnais, « il est prévu entre 39 et 41°C ».

En ce deuxième jour de la semaine, « les fortes chaleurs gagnent vers le nord et le nord-est : 36°C à 38°C sont attendus du Val-de-Loire à l’Île-de-France et au Grand-Est », prévient Météo-France.

Après une journée étouffante aujourd’hui, les températures « marquent un peu le pas en général au Sud » mercredi. Elles seront « en légère hausse dans le Nord-Est : des pointes à 40°C sont prévues en Bourgogne, encore autour de 35/36°C à Paris », a ajouté l’institut de prévision.
    """

    article2 = """
    La réforme du marché électrique européen adoptée en juillet 2024 prévoit une harmonisation au 1er janvier 2026 de l’heure limite de fermeture des échanges aux frontières via le marché infrajournalier, à 30 minutes avant le temps réel. Jusqu’à présent, RTE prenait la main dans sa « fenêtre opérationnelle » à partir de 60 minutes avant le temps réel.

Aux termes de la règlementation européenne, les gestionnaires de réseau de transport peuvent demander à leur régulateur national une dérogation allant jusqu’à trois années pour la mise en œuvre de cette évolution.

En considérant les analyses d’impacts et le plan d’action proposés par RTE, ainsi que les retours des acteurs de marché sur ces éléments, la CRE octroie à RTE une dérogation de trois ans pour la mise en œuvre de cette évolution, soit une réduction de la fenêtre opérationnelle à compter du 1er janvier 2029.

La mise en œuvre du plan d’action proposé par RTE nécessitera à la fois des évolutions des procédures internes de RTE et des modifications des règles de marché, qui devront être concertées avec la filière puis approuvées par la CRE.
    """

    article3 = """
     Infrastructure essentielle pour le pays, le réseau de transport d’électricité fait le trait d’union entre les sources de production et l’ensemble des consommateurs d’électricité. L’architecture du réseau actuel, colonne vertébrale du système électrique français, composé de lignes électriques très puissantes (400 000 volts) et de nombreuses lignes de différents niveaux de tensions inférieurs, constitue un atout national, adapté à une France dans laquelle l’électricité ne représente qu’à peine plus du quart de l’énergie consommée par le pays.

La France s’est fixé des objectifs ambitieux de décarbonation de son économie et de réindustrialisation qui doivent porter la part de l’électricité à plus de 50% dans notre mix énergétique de 2050. Cela rend nécessaire de renforcer le réseau public de transport d’électricité. A travers son plan stratégique d’investissements à l’horizon 2040 (SDDR), RTE propose ainsi une stratégie priorisée, optimisée et cadencée dans le temps pour mener à bien les transformations du réseau haute et très haute tension dans les 15 prochaines années.
Une stratégie de transformation du réseau électrique construite avec les territoires et les citoyens
[Vidéo] - Orientations de la stratégie de transformation du réseau de transport d’électricité à l’horizon 2040 (01:15)
Vignette de la vidéo Orientations de la stratégie de transformation du réseau de transport d’électricité à l’horizon 2040

Une concertation préalable à ces grandes orientations a été organisée avec l’ensemble des parties prenantes, de même qu’une consultation publique qui a permis de recueillir de nombreuses contributions. La moitié provient de collectivités territoriales ou d’aménageurs locaux. En cela, le SDDR constitue un plan d’aménagement national et territorial.

Les grandes orientations de ce plan d’investissement feront l’objet d’un débat public, organisé sous l’égide de la Commission nationale du débat public, et d’avis de l’Etat, de la Commission de régulation de l’énergie et de l’Autorité environnementale. A la suite de ces d’avis et de la participation du public, RTE publiera une version définitive de son plan d’investissements pour 2040, qui constituera sa stratégie de référence.
3 grands piliers pour un réseau performant et adapté à la décarbonation du pays

Le plan stratégique d’investissements s’articule autour de trois grands piliers stratégiques.
Renouveler le réseau et l’adapter au changement climatique, à un climat +4°C en 2100

Ces travaux représentent le principal programme industriel de ce plan.

Le réseau de transport d’électricité doit être pour partie renouvelé pour des questions d’âge. Il doit être également adapté pour faire face aux conséquences météorologiques du changement climatique. Au total 23 500 km de lignes, 85 000 pylônes et le système de télécom et contrôle commande seront renouvelés sur l’ensemble du territoire et dans tous les milieux (montagne, campagne, littoral, zones urbaines, etc.) pour un montant de l’ordre de 24 milliards d’euros.
[Vidéo] - Renouveler le réseau et l’adapter au changement climatique (01:12)
Vignette de la vidéo « Renouveler le réseau et l’adapter au changement climatique »
Raccorder la consommation d’électricité pour réussir l’électrification du pays et la réindustrialisation des territoires, et les nouvelles installations de production bas-carbone (renouvelables et nucléaire)

La décarbonation de l’industrie existante, l’accueil de nouveaux consommateurs (usines, datacenters, électrolyseurs) et le développement de moyens de production décarbonés sur l’ensemble du territoire (nucléaire, éolien en mer et renouvelables terrestres) occupent une place majeure dans ce programme industriel. 

Le SDDR projette de prioriser les infrastructures du réseau qui permettent de déclencher une électrification de l’économie.

Cette approche concerne, dans un premier temps, les sites industrialo-portuaires de Dunkerque, du Havre et de Fos-sur-Mer pour lesquels le niveau de maturité des projets est suffisant pour déclencher d’ores et déjà les investissements ; puis, 7 zones de développement économique (Saint-Avold, Sud Alsace, Vallée de la chimie, Plan-de-campagne, Loire-Estuaire, Sud Ile-de-France, Valenciennes) ainsi que d’autres zones issues du dialogue avec les collectivités territoriales, dans lesquelles les travaux seront lancés lorsque le niveau d’engagement des industriels sera avéré.

Du côté de la production d’électricité, le SDDR prévoit le raccordement des futurs EPR 2, projetés à l’horizon 2040, celui des énergies renouvelables en mer, en prévoyant la création d’un réseau de transport en mer qui n’existe pas aujourd’hui, ainsi que le raccordement des énergies renouvelables terrestres, sur la base des objectifs nationaux envisagés par l’État.

Ce pilier représente plus de la moitié des investissements nécessaires, soit 53 milliards d’€ pour des projets mis en service avant 2040.
[Vidéo] - Raccorder les nouveaux consommateurs et les nouvelles installations de production bas-carbone (01:53)
Vignette de la vidéo « Raccorder les nouveaux consommateurs et les nouvelles installations de production bas-carbone »
Renforcer la colonne vertébrale du réseau haute et très haute tension pour accueillir des flux d’électricité plus importants et répartis différemment sur le territoire, tout en limitant les congestions

D’ici 2040, l’enjeu est de faire transiter davantage d’électricité sur le réseau, tout en optimisant son fonctionnement afin d’éviter les congestions que pourraient générer ces nouveaux flux.

Pour cela, outre les travaux déjà engagés à l’horizon 2030, RTE identifie cinq grandes zones géographiques, à l’Ouest, à l’Est et au Sud de la France, dans lesquelles il sera prioritaire de renforcer le réseau très haute tension entre 2030 et 2040.

Afin d’augmenter les capacités techniques de la colonne vertébrale du réseau, RTE met en place une stratégie qui privilégie la transformation des infrastructures existantes ou leur doublement, dans leur tracé actuel.

Les nouvelles lignes très haute tension en dehors des tracés existants seront donc l’exception, et ne concerneront que les zones qui n’en comptent pas aujourd’hui ou dans lesquelles le maillage actuel est insuffisant.

Cette stratégie de renforcement, évaluée à 16,5 milliards d’euros, permet ainsi d’éviter la construction de 30% de lignes aériennes supplémentaires.
[Vidéo] - Renforcer la colonne vertébrale du réseau haute et très haute tension (02:26)
Vignette de la vidéo « Renforcer le réseau haute et très haute tension »
Un plan d’investissements qui s’adapte aux priorités publiques et au contexte macro-économique

De l’ordre de 100 milliards d’euros sur 15 ans, cette feuille de route industrielle doit permettre de réaliser les transformations nécessaires au fonctionnement du réseau de transport d’électricité, d’accompagner la décarbonation et la réindustrialisation de la France, tout en renforçant sa souveraineté.

Le SDDR inclut des analyses techniques, économiques et environnementales, et identifie la manière dont les différents scénarios et rythmes proposés pour le développement de l’infrastructure impactent ces paramètres. Pour la plupart des thèmes, il est séquencé en plusieurs périodes : jusqu’en 2030, 2030-2035 et 2035-2040.

Il ne conduit pas à engager aujourd’hui l’ensemble des investissements prévus : ceux-ci seront approuvés annuellement par la Commission de régulation de l’énergie.

L’évolution du besoin d’investissements en fonction de l’évolution des priorités publiques et du contexte macro-économique est connue et chiffrée.
Un levier pour la souveraineté qui mobilise le tissu industriel français et européen

Ce plan a pour ambition de maximiser les retombées économiques en France et en Europe.

Au-delà de permettre le développement économique des territoires, en facilitant l’accueil de nouveaux consommateurs (usines, data centers, électrolyseurs, etc.), le développement du réseau de transport d’électricité nécessite d’organiser et de mobiliser une base industrielle manufacturières, majoritairement installée en Europe (câbles, postes en mer, transformateurs, etc.).

RTE a déjà incité plusieurs fournisseurs à investir en France : à l’exemple des Chantiers de l’Atlantique qui fabriqueront à Saint-Nazaire les trois premières plateformes en mer et trois stations de conversions à terre françaises à courant continu ou encore le câblier italien Prysmian qui développera une nouvelle ligne de production dans son usine de Montereau-Fault-Yonne (Seine-et-Marne).

En outre, dans un contexte de fortes tensions pour la fourniture des matériels nécessaires aux réseaux, RTE a lancé un appel à manifestation d’intérêt auprès des équipementiers en vue d’identifier les conditions pour l’implantation d’une usine de production de câbles sous-marins en France pour couvrir notamment les besoins nationaux ; le pays n’en étant aujourd’hui pas doté.

En matière d’emplois, enfin, la croissance des investissements dans les réseaux de distribution et de transport d’électricité se traduit par de nouveaux débouchés significatifs en France : près de 8 000 à 12 000 emplois supplémentaires pourraient être créés par an d’ici 2030 à l’échelle de la filière (gestionnaires de réseaux, fournisseurs, prestataires).
    """

    article4 = """
     En bref : Sam Altman a dévoilé GPT-5, un modèle d'intelligence artificielle présenté comme une avancée majeure vers l'intelligence artificielle générale, offrant des capacités d'expert de niveau doctorat. Ce modèle, plus rapide et plus précis que ses prédécesseurs, est désormais le modèle par défaut de ChatGPT, accessible gratuitement.
Sommaire
A l'usage : plus rapide, plus de contexte et beaucoup moins d'hallucinations
Rationalisation : GPT-5, nouveau modèle par défaut tout-en-un
Performances
Une fiabilité encore loin d'être à toute épreuve
Comme il l'avait laissé entendre en début de semaine, Sam Altman a dévoilé hier en fin de journée GPT-5. Le modèle, qu'il présente comme une avancée majeure et une étape significative vers l'AGI (une promesse à manier avec prudence), l'intelligence artificielle générale, aurait les capacités d'un expert de niveau doctorat. Cerise sur le gâteau : il est accessible aux utilisateurs gratuits puisqu'il est désormais le modèle par défaut de ChatGPT.
 
A l'usage : plus rapide, plus de contexte et beaucoup moins d'hallucinations
Difficile de provoquer encore un "effet wouaw" : en façade, les précédentes versions de GPT semblaient être capables de tout faire, avec un aplomb certain et une capacité de conviction capable d'en éblouir plus d'un. Si bien que Sam Altman a dû multiplier les déclarations ces derniers jours pour faire passer le message que GPT-5 est une nouvelle avancée majeure.
La première évidence, c'est la rapidité de production de contenu de GPT-5, encore plus élevée que celle des précédentes versions. Aussitôt mis à disposition, nous avons réalisé un test de développement : GPT-5 a réglé en 1 minute un problème de conflit de frameworks sur lequel GPT-4o et Claude 4 Sonnet tournaient en rond. Les parts de marché auprès des développeurs semble d'ailleurs être une des priorités d'OpenAI, en témoigne le partenariat avec Cursor et la mise à disposition gratuite auprès des ses utilisateurs de GPT-5 pendant la phase de lancement.
Rationalisation : GPT-5, nouveau modèle par défaut tout-en-un
Sur le papier, GPT-5 combine un modèle rapide pour les questions simples, un modèle de raisonnement profond pour les problèmes complexes, et un routeur intelligent qui choisit lequel utiliser en fonction du type de conversation. Un bon moyen de simplifier les choses pour les utilisateurs et de rationaliser les coûts.
Comme pour GPT-4o, la différence entre l’accès gratuit et payant à GPT-5 au sein de ChatGPT repose sur le volume d’utilisation. Lorsque les utilisateurs gratuits ont atteint leur quota, ils sont automatiquement redirigés vers GPT-5 mini, un modèle allégé mais très performant, selon OpenAI. La limite d'utilisation est nettement plus élevée pour les abonnés Plus tandis que ceux de Pro ont un accès illimité à GPT-5 et peuvent activer GPT-5 Pro, une version dont les capacités de raisonnement ont été étendues.
GPT‑5 est également disponible via l’API de la société. Trois variantes sont proposées aux développeurs : gpt‑5, gpt‑5‑mini et gpt‑5‑nano, permettant d’équilibrer performances, coûts et latence.
Performances
Le modèle bénéficie d’une fenêtre de contexte élargie à 256 000 tokens, lui permettant de traiter des documents volumineux ou de suivre des échanges longs sans perte de cohérence. Il est non seulement plus rapide que ses prédécesseurs, mais son taux d’hallucination aurait été significativement réduit, renforçant la fiabilité de ses réponses.
Selon OpenAI, il établit un nouvel état de l’art dans les domaines des mathématiques (94,6 % sur AIME 2025 sans outils), du codage du monde réel (74,9 % sur SWE-bench Verified, 88 % sur Aider Polyglot), de la compréhension multimodale (84,2 % sur MMMU) et de la santé (46,2 % sur HealthBench Hard).
Côté sécurité, le modèle a été rigoureusement testé à travers 5 000 heures de red teaming en collaboration avec des organismes spécialisés tels que le CAISI et l’AISI britannique. L'entreprise a mis en place des mesures de protection robustes :
"Bien que nous n’ayons pas de preuves définitives que ce modèle pourrait aider de manière significative un novice à créer de graves dommages biologiques, notre seuil défini pour une capacité élevée, nous adoptons une approche de précaution et nous activons dès maintenant les mesures de protection requises afin d’être plus prêts lorsque de telles capacités seront disponibles".
Microsoft a d'ores et déjà intégré GPT-5 à la plupart de ses produits : Copilot, Microsoft 365 Copilot (Word, Excel, Outlook...), GitHub Copilot, Visual Studio Code ou Azure AI Foundry.
Une fiabilité encore loin d'être à toute épreuve

Sur le papier, les promesses de réduction des hallucinations semblent être l'une des plus belles améliorations de GPT-5. Mais dans les faits, il n'a pas fallu 5 minutes pour induire le nouveau modèle phare d'OpenAI en erreur. Or, si le modèle se trompe sur le président des Etats-Unis, il y a fort à parier que les réponses soient encore truffées d'erreurs sur des questions plus spécifiques.
 

    """
    l = [article, article2, article3, article4]
    l = l * 10
    results = asyncio.run(score_articles(l))
    print(results)
