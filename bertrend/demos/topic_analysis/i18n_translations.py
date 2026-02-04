#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
# Translation dictionaries for topic_analysis demo
TRANSLATIONS = {
    "app_title": {
        "fr": "BERTrend - Démo d'analyse de sujets",
        "en": "BERTrend - Topic Analysis demo",
    },
    "data_distribution": {
        "fr": "Distribution des données",
        "en": "Data distribution",
    },
    "time_aggregation": {
        "fr": "Agrégation temporelle",
        "en": "Time aggregation",
    },
    "save_model": {
        "fr": "Sauvegarder le modèle",
        "en": "Save Model",
    },
    "enter_model_name": {
        "fr": "Entrez un nom pour le modèle (optionnel) :",
        "en": "Enter a name for the model (optional):",
    },
    "model_saved_successfully": {
        "fr": "Modèle sauvegardé avec succès sous {model_save_path}",
        "en": "Model saved successfully as {model_save_path}",
    },
    "training_model": {
        "fr": "Entraînement du modèle...",
        "en": "Training model...",
    },
    "train_model": {
        "fr": "Entraîner le modèle",
        "en": "Train Model",
    },
    "review_settings_help": {
        "fr": "Assurez-vous de vérifier les paramètres avant de cliquer sur ce bouton.",
        "en": "Make sure to review the settings before clicking on this button.",
    },
    "error_embedding_documents": {
        "fr": "Une erreur s'est produite lors de la vectorisation des documents : {e}",
        "en": "An error occurred while embedding documents: {e}",
    },
    "topic_analysis_demo_title": {
        "fr": ":part_alternation_mark: Démo d'analyse de sujets",
        "en": ":part_alternation_mark: Topic analysis demo",
    },
    "data_loading_training": {
        "fr": "Chargement des données & entraînement du modèle",
        "en": "Data loading & model training",
    },
    "topic_exploration": {
        "fr": "Exploration des sujets",
        "en": "Topic exploration",
    },
    "topic_visualization": {
        "fr": "Visualisation des sujets",
        "en": "Topic visualization",
    },
    "temporal_visualization": {
        "fr": "Visualisation temporelle",
        "en": "Temporal visualization",
    },
    "newsletter_generation": {
        "fr": "Génération de newsletter",
        "en": "Newsletter generation",
    },
    "topic_analysis": {
        "fr": "Analyse de sujets",
        "en": "Topic Analysis",
    },
    "application_example": {
        "fr": "Exemple d'application",
        "en": "Application example",
    },
    "embeddings_cache_info": {
        "fr": "Les embeddings ne sont pas sauvegardés dans le cache et ne sont donc pas chargés. Veuillez vous assurer d'entraîner le modèle sans utiliser les embeddings en cache si vous souhaitez des visualisations temporelles correctes et fonctionnelles.",
        "en": "Embeddings aren't saved in cache and thus aren't loaded. Please make sure to train the model without using cached embeddings if you want correct and functional temporal visualizations.",
    },
    "save_model_reminder": {
        "fr": "N'oubliez pas de sauvegarder votre modèle !",
        "en": "Don't forget to save your model!",
    },
    "no_model_available_error": {
        "fr": "Aucun modèle disponible à sauvegarder. Veuillez d'abord entraîner un modèle.",
        "en": "No model available to save. Please train a model first.",
    },
    "no_document_for_topic": {
        "fr": "Aucun document trouvé pour le sujet sélectionné.",
        "en": "No documents found for the selected topic.",
    },
    "train_model_first_error": {
        "fr": "Veuillez d'abord entraîner un modèle.",
        "en": "Please train a model first.",
    },
    "remote_embedding_service_type_not_supported_error": {
        "fr": "Ces visualisations ne sont disponibles que si un service d'embedding local est utilisé.",
        "en": "These visualizations are only available if a local embedding service is used.",
    },
    "visualizations": {
        "fr": "Visualisations",
        "en": "Visualizations",
    },
    "include_outliers": {
        "en": "Include outliers (Topic = -1)",
        "fr": "Inclure les outliers (Topic = -1)",
    },
    "overall_results": {
        "en": "Overall results",
        "fr": "Aperçu des résultats",
    },
    "overall_results_display_error": {
        "en": "Cannot display overall results",
        "fr": "Impossible d'afficher l'aperçu des résultats",
    },
    "change_umap_params_warning": {
        "en": "Try to change the UMAP parameters",
        "fr": "Essayez de changer les paramètres UMAP",
    },
    "topics_treemap": {"en": "Topics Treemap", "fr": "Topics Treemap"},
    "topics_treemap_computation": {
        "en": "Computing topics treemap...",
        "fr": "Calcul du treemap des sujets...",
    },
    "data_map": {"en": "Data Map", "fr": "Carte des données"},
    "data_map_loading": {
        "en": "Loading Data-map plot...",
        "fr": "Chargement de la carte des données...",
    },
    "full_screen": {"en": "Full screen", "fr": "Affichage plein écran"},
    "no_data_map_warning": {
        "en": "No valid topics to visualize. All documents might be classified as outliers.",
        "fr": "Pas de sujets à visualiser. Peut-être trop d'outliers.",
    },
    # TEMPTopic Parameters section
    "temptopic_parameters": {
        "en": "TEMPTopic Parameters",
        "fr": "Paramètres TEMPTopic",
    },
    "window_size": {"en": "Window Size", "fr": "Taille de fenêtre"},
    "k_nearest_embeddings": {
        "en": "Number of Nearest Embeddings (k)",
        "fr": "Nombre d'Embeddings les plus proches (k)",
    },
    "k_nearest_help": {
        "en": "The k-th nearest neighbor used for Topic Representation Stability calculation.",
        "fr": "Le k-ième voisin le plus proche utilisé pour le calcul de la stabilité de représentation des sujets.",
    },
    "alpha_weight": {
        "en": "Alpha (Topic vs Representation Stability Weight)",
        "fr": "Alpha (Poids de Stabilité du Sujet vs Représentation)",
    },
    "alpha_help": {
        "en": "Closer to 1 gives more weight given to Topic Embedding Stability, Closer to 0 gives more weight to topic representation stability.",
        "fr": "Plus proche de 1 donne plus de poids à la stabilité de l'embedding du sujet, plus proche de 0 donne plus de poids à la stabilité de la représentation du sujet.",
    },
    "use_double_agg": {
        "en": "Use Double Aggregation",
        "fr": "Utiliser l'Agrégation Double",
    },
    "double_agg_help": {
        "en": "If unchecked, only Document Aggregation Method will be globally used.",
        "fr": "Si non coché, seule la méthode d'agrégation de documents sera utilisée globalement.",
    },
    "doc_agg_method": {
        "en": "Document Aggregation Method",
        "fr": "Méthode d'Agrégation de Documents",
    },
    "global_agg_method": {
        "en": "Global Aggregation Method",
        "fr": "Méthode d'Agrégation Globale",
    },
    "use_evolution_tuning": {
        "en": "Use Evolution Tuning",
        "fr": "Utiliser Evolution Tuning",
    },
    "use_global_tuning": {
        "en": "Use Global Tuning",
        "fr": "Utiliser Global Tuning",
    },
    # Time granularity section
    "select_time_granularity": {
        "en": "Select custom time granularity",
        "fr": "Sélectionner la granularité temporelle personnalisée",
    },
    "days": {"en": "Days", "fr": "Jours"},
    "hours": {"en": "Hours", "fr": "Heures"},
    "minutes": {"en": "Minutes", "fr": "Minutes"},
    "seconds": {"en": "Seconds", "fr": "Secondes"},
    "granularity_info": {
        "en": "Granularity must be greater than zero and less than or equal to {max_granularity}.",
        "fr": "La granularité doit être supérieure à zéro et inférieure ou égale à {max_granularity}.",
    },
    # Buttons and controls
    "apply_granularity": {
        "en": "Apply Granularity and Parameters",
        "fr": "Appliquer la Granularité et les Paramètres",
    },
    "show_table_results": {
        "en": "Show table results",
        "fr": "Afficher les résultats du tableau",
    },
    "topic_evolution": {"en": "Topic evolution", "fr": "Évolution des sujets"},
    "topic_info": {"en": "Topic info", "fr": "Informations sur les sujets"},
    "documents_per_date": {"en": "Documents per date", "fr": "Documents par date"},
    # Dataframes and expandable sections
    "topic_evolution_dataframe": {
        "en": "Topic Evolution Dataframe",
        "fr": "Dataframe d'évolution des sujets",
    },
    "topic_info_dataframe": {
        "en": "Topic Info Dataframe",
        "fr": "Dataframe d'informations sur les sujets",
    },
    "documents_per_date_dataframe": {
        "en": "Documents per Date Dataframe",
        "fr": "Dataframe des documents par date",
    },
    "temptopic_visualizations": {
        "en": "TempTopic Visualizations",
        "fr": "Visualisations TempTopic",
    },
    # Visualization settings
    "topics_to_show": {"en": "Topics to Show", "fr": "Sujets à Afficher"},
    "topic_evolution_header": {
        "en": "Topic Evolution in Time and Semantic Space",
        "fr": "Évolution des Sujets dans le temps et l'espace sémantique",
    },
    "umap_n_neighbors": {"en": "UMAP n_neighbors", "fr": "UMAP n_neighbors"},
    "umap_min_dist": {"en": "UMAP min_dist", "fr": "UMAP min_dist"},
    "umap_metric": {"en": "UMAP Metric", "fr": "UMAP Metric"},
    "color_palette": {"en": "Color Palette", "fr": "Palette de Couleurs"},
    "overall_topic_stability": {
        "en": "Overall Topic Stability",
        "fr": "Stabilité globale des sujets",
    },
    "normalize": {"en": "Normalize", "fr": "Normaliser"},
    "temporal_stability_metrics": {
        "en": "Temporal Stability Metrics",
        "fr": "Métriques de stabilité temporelle",
    },
    # Topic popularity
    "popularity_of_topics": {
        "en": "Popularity of topics over time",
        "fr": "Popularité des sujets au fil du temps",
    },
    "topics_list_format": {
        "en": "Topics list (format 1,12,52 or 1:20)",
        "fr": "Liste des sujets (format 1,12,52 ou 1:20)",
    },
    "nr_bins": {"en": "nr_bins", "fr": "nombre_de_bins"},
    # Messages
    "apply_granularity_message": {
        "en": "Please apply granularity and parameters to view the temporal visualizations.",
        "fr": "Veuillez appliquer la granularité et les paramètres pour voir les visualisations temporelles.",
    },
    "fitting_temptopic": {
        "en": "Fitting TempTopic...",
        "fr": "Fitting de TempTopic...",
    },
    "computing_topics": {
        "en": "Computing topics over time...",
        "fr": "Calcul des sujets au fil du temps...",
    },
    "select_valid_granularity": {
        "en": "Please select a valid granularity before applying.",
        "fr": "Veuillez sélectionner une granularité valide avant d'appliquer.",
    },
    "temporal_visualizations": {
        "en": "Temporal visualizations of topics",
        "fr": "Visualisations temporelles des sujets",
    },
    # New translations for newsletter functionality
    "newsletter_generation_title": {
        "en": "Automatic newsletters generation",
        "fr": "Génération automatique de newsletters",
    },
    "include_all_topics": {"en": "Include all topics", "fr": "Inclure tous les sujets"},
    "number_of_topics": {"en": "Number of topics", "fr": "Nombre de sujets"},
    "include_all_documents": {
        "en": "Include all documents per topic",
        "fr": "Inclure tous les documents par sujet",
    },
    "number_of_docs_per_topic": {
        "en": "Number of docs per topic",
        "fr": "Nombre de documents par sujet",
    },
    "improve_topic_description": {
        "en": "Improve topic description",
        "fr": "Améliorer la description du sujet",
    },
    "summary_mode": {"en": "Summary mode", "fr": "Mode de résumé"},
    "summarizer_class": {
        "en": "Summarizer class",
        "fr": "Classe utilisée pour le résumé",
    },
    "generate_newsletter_button": {
        "en": "Generate newsletter",
        "fr": "Générer la newsletter",
    },
    "generating_newsletters": {
        "en": "Generating newsletters...",
        "fr": "Génération des newsletters...",
    },
    # New translations for topic exploration functionality
    "topics_exploration": {"en": "Topics exploration", "fr": "Exploration des sujets"},
    "search_topic": {"en": "Search topic", "fr": "Rechercher un sujet"},
    "topic": {"en": "Topic", "fr": "Sujet"},
    "documents_lowercase": {"en": "documents", "fr": "documents"},
    "unknown_source": {"en": "Unknown Source", "fr": "Source Inconnue"},
    "new_documents": {"en": "New documents", "fr": "Nouveaux documents"},
    "generate_topic_description": {
        "en": "Generate a short description of the topic",
        "fr": "Générer une description courte du sujet",
    },
    "generating_description": {
        "en": "Generating the description...",
        "fr": "Génération de la description en cours...",
    },
    "number_of_articles_to_display": {
        "en": "Number of articles to display",
        "fr": "Nombre d'articles à afficher",
    },
    "select_sources_to_display": {
        "en": "Select the sources to display",
        "fr": "Sélectionner les sources à afficher",
    },
    "all": {"en": "All", "fr": "Toutes"},
    "export_configuration": {
        "en": "Export Configuration",
        "fr": "Configuration de l'export",
    },
    "choose_export_method": {
        "en": "Choose export method:",
        "fr": "Choisir la méthode d'export :",
    },
    "download_as_zip": {"en": "Download as ZIP", "fr": "Télécharger en ZIP"},
    "save_to_folder": {"en": "Save to folder", "fr": "Enregistrer dans un dossier"},
    "export_method_help": {
        "en": "Select whether to download documents as a ZIP file or save them directly to a folder on the server.",
        "fr": "Sélectionnez si vous souhaitez télécharger les documents sous forme de fichier ZIP ou les enregistrer directement dans un dossier sur le serveur.",
    },
    "granularity_days": {
        "en": "Granularity (number of days)",
        "fr": "Granularité (nombre de jours)",
    },
    "export_topic_documents": {
        "en": "Export Topic Documents",
        "fr": "Exporter les Documents associés au Sujet",
    },
    "export_success": {
        "en": "Successfully exported documents to folder: {export_folder}",
        "fr": "Documents exportés avec succès dans le dossier : {export_folder}",
    },
}
