# Architecture simplifiée de prospective_demo

```mermaid
graph TB
    subgraph Frontend["Frontend (Streamlit)"]
        APP[app.py<br/>Interface utilisateur Streamlit]
        DASH[Dashboards<br/>- Analyse<br/>- Signaux<br/>- Rapports<br/>- Comparatif]
        CONFIG[Configuration<br/>- Sources d'information<br/>- Modèles]
    end

    subgraph Backend["Backend (FastAPI)"]
        API[bertrend_service.py<br/>API REST]
        ROUTERS[Routers<br/>- Data Provider<br/>- Newsletters<br/>- BERTrend App<br/>- Info]
        PROCESS[process_new_data.py<br/>Traitement des données<br/>- Chargement<br/>- Analyse<br/>- ML/LLM]
    end

    subgraph Data["Data Layer"]
        FEEDS[Feeds<br/>Flux de données JSONL]
        MODELS[Models<br/>Modèles BERTrend<br/>BERTopic]
        REPORTS[Reports<br/>Rapports générés]
        DB[(Base de données<br/>Articles, Topics,<br/>Signaux)]
    end

    subgraph Scheduling["Scheduling Service"]
        SCHED[scheduling_service.py<br/>Service de planification]
        JOBS[Jobs<br/>- Collecte de données<br/>- Entraînement modèles<br/>- Génération rapports]
        CRON[Scheduler APScheduler]
    end

    %% Interactions Frontend
    APP --> |API REST| API
    APP --> |Lecture| MODELS
    APP --> |Lecture| REPORTS
    
    %% Interactions Backend
    API --> |CRUD| DB
    API --> |Accès| FEEDS
    PROCESS --> |Lecture| FEEDS
    PROCESS --> |Écriture| MODELS
    PROCESS --> |Écriture| DB
    PROCESS --> |Génération| REPORTS
    
    %% Interactions Scheduling
    SCHED --> |Déclenche| PROCESS
    SCHED --> |Appelle API| API
    JOBS --> |Planification| CRON
    SCHED --> |Accès| FEEDS
    
    %% Style
    classDef frontend fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef backend fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef data fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef scheduling fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    
    class Frontend,APP,DASH,CONFIG frontend
    class Backend,API,ROUTERS,PROCESS backend
    class Data,FEEDS,MODELS,REPORTS,DB data
    class Scheduling,SCHED,JOBS,CRON scheduling
```

## Description des composants

### Frontend (Streamlit)
- **app.py** : Point d'entrée de l'application Streamlit avec authentification
- **Dashboards** : Interfaces utilisateur pour l'analyse, la visualisation des signaux, génération de rapports et comparaisons
- **Configuration** : Interfaces de configuration des sources de données et des modèles

### Backend (FastAPI)
- **bertrend_service.py** : Service API REST FastAPI principal
- **Routers** : Endpoints pour la gestion des données, newsletters, et applications BERTrend
- **process_new_data.py** : Pipeline de traitement des données incluant ML/LLM

### Data Layer
- **Feeds** : Stockage des flux de données au format JSONL
- **Models** : Stockage des modèles entraînés (BERTrend, BERTopic)
- **Reports** : Rapports générés et analyses
- **Base de données** : Stockage structuré des articles, topics et signaux

### Scheduling Service
- **scheduling_service.py** : Service de planification indépendant
- **Jobs** : Tâches planifiées (collecte, entraînement, génération)
- **Scheduler** : APScheduler pour l'exécution planifiée des tâches
