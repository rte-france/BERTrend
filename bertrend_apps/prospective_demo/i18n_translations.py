#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Translation dictionaries for prospective_demo
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
        "en": "Model Status by Monitoring Feed",
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
    "select_feed": {"fr": "Sélection de la veille", "en": "Select monitored feed"},
    "no_available_model_warning": {
        "fr": "Pas de modèle disponible",
        "en": "No model available",
    },
    "at_least_2models_warning": {
        "fr": "2 modèles minimum pour analyser les tendances !",
        "en": "At least 2 models are required for trend analysis!",
    },
    "analysis_date": {"fr": "Date d'analyse", "en": "Analysis Date"},
    "analysis_date_help": {
        "fr": "Sélection de la date d'analyse parmi celles disponibles",
        "en": "Selection of the analysis date from those available",
    },
    # dashboard_signals.py translations
    "title": {"fr": "Titre", "en": "Title"},
    "todo_message": {
        "fr": "TODO",
        "en": "TODO",
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
    "untitled_topic": {"fr": "???Titre???", "en": "???Title???"},
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
    "no_data_for_signal": {
        "en": "No data found for signal ID: {signal_id}",
        "fr": "Aucune donnée trouvée pour l'identifiant de signal : {signal_id}",
    },
    # Feed configuration dialog
    "feed_config_dialog_title": {
        "fr": "Configuration d'un nouveau flux de données",
        "en": "New Data Feed Configuration",
    },
    # Form labels
    "feed_id_label": {
        "fr": "ID",
        "en": "ID",
    },
    "feed_id_help": {
        "fr": "Identifiant du flux de données",
        "en": "Data feed identifier",
    },
    "feed_source_label": {
        "fr": "Source",
        "en": "Source",
    },
    "feed_source_help": {
        "fr": "Sélection de la source de données",
        "en": "Data source selection",
    },
    "feed_query_label": {
        "fr": "Requête",
        "en": "Query",
    },
    "feed_query_help": {
        "fr": "Saisir ici la requête qui sera faite sur Google News",
        "en": "Enter the query that will be made on Google News",
    },
    "feed_language_label": {
        "fr": "Langue",
        "en": "Language",
    },
    "feed_language_help": {
        "fr": "Choix de la langue",
        "en": "Language selection",
    },
    "feed_frequency_label": {
        "fr": "Fréquence d'exécution",
        "en": "Execution Frequency",
    },
    "feed_frequency_help": {
        "fr": "Fréquence de collecte des données",
        "en": "Data collection frequency",
    },
    "feed_atom_label": {
        "fr": "ATOM feed",
        "en": "ATOM feed",
    },
    "feed_atom_help": {
        "fr": "URL du flux de données ATOM",
        "en": "ATOM data feed URL",
    },
    # Language options
    "language_english": {
        "fr": "Anglais",
        "en": "English",
    },
    "language_french": {
        "fr": "Français",
        "en": "French",
    },
    # Buttons
    "ok_button": {
        "fr": "OK",
        "en": "OK",
    },
    "yes_button": {
        "fr": "Oui",
        "en": "Yes",
    },
    "no_button": {
        "fr": "Non",
        "en": "No",
    },
    # Help texts
    "new_feed_help": {
        "fr": "Nouveau flux de veille",
        "en": "New monitoring feed",
    },
    # Error messages
    "cron_error_message": {
        "fr": "Expression mal écrite !",
        "en": "Badly written expression!",
    },
    # Toast messages
    "feed_deactivated_message": {
        "fr": "Le flux **{feed_id}** est déactivé !",
        "en": "Feed **{feed_id}** is deactivated!",
    },
    "feed_activated_message": {
        "fr": "Le flux **{feed_id}** est activé !",
        "en": "Feed **{feed_id}** is activated!",
    },
    # Dialog titles
    "confirmation_dialog_title": {
        "fr": "Confirmation",
        "en": "Confirmation",
    },
    # Confirmation messages
    "delete_feed_confirmation": {
        "fr": "Voulez-vous vraiment supprimer le flux de veille **{feed_id}** ?",
        "en": "Do you really want to delete the monitoring feed **{feed_id}**?",
    },
    "deactivate_feed_confirmation": {
        "fr": "Voulez-vous vraiment désactiver le flux de veille **{feed_id}** ?",
        "en": "Do you really want to deactivate the monitoring feed **{feed_id}**?",
    },
    "activate_feed_message": {
        "fr": "Activation du flux de veille **{feed_id}**",
        "en": "Activating monitoring feed **{feed_id}**",
    },
    # Form labels
    "monitoring_selection_label": {
        "fr": "Sélection de la veille",
        "en": "Monitoring Selection",
    },
    "time_window_label": {
        "fr": "Fenêtre temporelle (jours)",
        "en": "Time Window (days)",
    },
    # Statistics table labels
    "stats_id_label": {
        "fr": "ID",
        "en": "ID",
    },
    "stats_files_count_label": {
        "fr": "# Fichiers",
        "en": "# Files",
    },
    "stats_start_date_label": {
        "fr": "Date début",
        "en": "Start Date",
    },
    "stats_end_date_label": {
        "fr": "Date fin",
        "en": "End Date",
    },
    "stats_articles_count_label": {
        "fr": "# Articles",
        "en": "# Articles",
    },
    "stats_recent_articles_count_label": {
        "fr": "# Articles ({days} derniers jours)",
        "en": "# Articles (last {days} days)",
    },
    # Section titles
    "recent_data_title": {
        "fr": "Données des derniers {days} jours",
        "en": "Data from the last {days} days",
    },
    # Column headers
    "col_id": {"fr": "id", "en": "id"},
    "col_num_models": {"fr": "# modèles", "en": "# models"},
    "col_first_model_date": {"fr": "date 1er modèle", "en": "first model date"},
    "col_last_model_date": {"fr": "date dernier modèle", "en": "last model date"},
    "col_update_frequency": {
        "fr": "fréquence mise à jour (# jours)",
        "en": "update frequency (# days)",
    },
    "col_analysis_window": {
        "fr": "fenêtre d'analyse (# jours)",
        "en": "analysis window (# days)",
    },
    # Dialog titles and messages
    "dialog_parameters": {"fr": "Paramètres", "en": "Parameters"},
    "dialog_confirmation": {"fr": "Confirmation", "en": "Confirmation"},
    "dialog_model_regeneration": {
        "fr": "Regénération des modèles",
        "en": "Model Regeneration",
    },
    # Model parameters
    "model_params_title": {
        "fr": "Paramètres des modèles pour la veille {}",
        "en": "Model parameters for monitoring {}",
    },
    "update_frequency_label": {
        "fr": "Fréquence de mise à jour des modèles (en jours)",
        "en": "Model update frequency (in days)",
    },
    "update_frequency_help": {
        "fr": "Sélection de la fréquence à laquelle la détection de sujets est effectuée. Le nombre de jours sélectionné doit être choisi pour s'assurer d'un volume de données suffisant.",
        "en": "Selection of the frequency at which topic detection is performed. The number of days selected should be chosen to ensure sufficient data volume.",
    },
    "time_window_help": {
        "fr": "Sélection de la plage temporelle considérée pour calculer les différents types de signaux (faibles, forts)",
        "en": "Selection of the time range considered to calculate different types of signals (weak, strong)",
    },
    # Analysis parameters
    "analysis_params_title": {
        "fr": "Paramètres d'analyse de la veille {}: éléments à inclure",
        "en": "Analysis parameters for monitoring {}: elements to include",
    },
    "topic_evolution": {"fr": "Evolution du sujet", "en": "Topic evolution"},
    "evolution_scenarios": {"fr": "Scénarios d'évolution", "en": "Evolution scenarios"},
    "multifactorial_analysis": {
        "fr": "Analyse multifactorielle",
        "en": "Multifactorial analysis",
    },
    # Buttons and actions
    "btn_ok": {"fr": "OK", "en": "OK"},
    "btn_yes": {"fr": "Oui", "en": "Yes"},
    "btn_no": {"fr": "Non", "en": "No"},
    # Delete confirmation
    "delete_models_warning": {
        "fr": "Voulez-vous vraiment supprimer tous les modèles stockés pour la veille **{}** ?",
        "en": "Do you really want to delete all stored models for monitoring **{}**?",
    },
    "models_deleted_success": {
        "fr": "Modèles en cache supprimés pour la veille {} !",
        "en": "Cached models deleted for monitoring {}!",
    },
    # Regeneration
    "regenerate_models_warning": {
        "fr": "Voulez-vous re-générer l'ensemble des modèles pour la veille {} ?",
        "en": "Do you want to regenerate all models for monitoring {}?",
    },
    "regenerate_models_delete_warning": {
        "fr": "L'ensemble des modèles existant pour cette veille sera supprimé.",
        "en": "All existing models for this monitoring will be deleted.",
    },
    "regenerate_models_irreversible": {
        "fr": "Attention, cette regénération ne peut pas être annulée une fois lancée !",
        "en": "Warning, this regeneration cannot be cancelled once started!",
    },
    "regeneration_in_progress": {
        "fr": "Regénération en cours des modèles pour la veille {}. L'opération peut prendre un peu de temps.",
        "en": "Model regeneration in progress for monitoring {}. The operation may take some time.",
    },
    "regeneration_close_info": {
        "fr": "Vous pouvez fermer cette fenêtre.",
        "en": "You can close this window.",
    },
    # Learning toggle
    "learning_deactivated": {
        "fr": "Le learning pour la veille **{}** est déactivé !",
        "en": "Learning for monitoring **{}** is deactivated!",
    },
    "learning_activated": {
        "fr": "Le learning pour la veille **{}** est activé !",
        "en": "Learning for monitoring **{}** is activated!",
    },
    "deactivate_learning_question": {
        "fr": "Voulez-vous vraiment l'apprentissage pour le flux de veille **{}** ?",
        "en": "Do you really want to deactivate learning for monitoring feed **{}**?",
    },
    "activate_learning_info": {
        "fr": "Activation de l'apprentissage pour le flux de veille **{}",
        "en": "Activating learning for monitoring feed **{}**",
    },
    # Titles and Steps
    "step_1_title": {
        "fr": "Etape 1: Sélection des sujets à retenir",
        "en": "Step 1: Choose topics to keep",
    },
    "step_2_title": {
        "fr": "Etape 2: Configuration de l'export",
        "en": "Step 2: Export Configuration",
    },
    # Messages
    "generate_button_label": {"fr": "Générer", "en": "Generate"},
    "download_button_label": {"fr": "Télécharger (html)", "en": "Download (html)"},
    "download_json_button_label": {"fr": "Télécharger (json)", "en": "Download (json)"},
    "send_button_label": {"fr": "Envoyer", "en": "Send"},
    # Error Messages
    "invalid_email": {"fr": "Adresse email incorrecte", "en": "Invalid email address"},
    "email_error_message": {
        "fr": "Erreur lors de l'envoi de l'email",
        "en": "Error sending email",
    },
    # Success Messages
    "email_sent_successfully": {
        "fr": "Email envoyé avec succès!",
        "en": "Email sent successfully!",
    },
    # Report
    "report_title_part_1": {"fr": "Actu", "en": "Actu"},
    "report_mail_title": {"fr": "Rapport veille", "en": "Monitoring report"},
    "email_recipients": {"fr": "Destinataires", "en": "Recipients"},
    "report_preview_title": {"fr": "Rapport (aperçu)", "en": "Report (Preview)"},
    # Miscellaneous
    "split_by_paragraph": {
        "en": "Split text by paragraphs for analysis",
        "fr": "Découper le texte par paragraphes pour l'analyse",
    },
    "split_by_paragraph_help": {
        "en": "Split text by paragraphs for analysis (useful for long text such as news articles which may contain different subtopics)",
        "fr": "Découpe le texte par paragraphes pour l'analyse (utile pour des textes longs comme des articles de presse qui peuvent contenir plusieurs sous-sujets)",
    },
    "evaluate_articles_quality": {
        "en": "Keep only high-quality articles",
        "fr": "Retenir seulement les articles de meilleure qualité",
    },
    "evaluate_articles_quality_help": {
        "en": "Score each article using a LLM-based approach ($$) and keep only high-quality articles",
        "fr": "Evalue chaque article avec une approche basée sur des LLM ($$) et ne retient que les articles de meilleure qualité",
    },
}
