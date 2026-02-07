# BERTrend Architecture Documentation

This document provides architectural diagrams for the BERTrend system, showing the main components and their interactions.

## System Architecture

```mermaid
graph TD
    User[User] --> Apps[BERTrend Apps]
    Apps --> Core[BERTrend Core]

    subgraph CoreComponents[Core Components]
        BERTrend[BERTrend Class] --> TopicModel[BERTopicModel]
        BERTrend --> EmbeddingService[Embedding Service]
        BERTrend --> SignalAnalysis[Signal Analysis]
        TopicModel --> BERTopic[BERTopic Library]
    end

    subgraph Services[Services]
        EmbeddingService --> LocalEmbedding[Local Embedding]
        EmbeddingService --> RemoteEmbedding[Remote Embedding API]
        LocalEmbedding --> SentenceTransformer[Sentence Transformer]
    end

    subgraph Analysis[Analysis Components]
        SignalAnalysis --> WeakSignals[Weak Signals Detection]
        SignalAnalysis --> TopicClassification[Topic Classification]
    end

    Core --> CoreComponents
    CoreComponents --> Services
    CoreComponents --> Analysis
```

## Component Interactions

```mermaid
sequenceDiagram
    participant User
    participant App as BERTrend App
    participant Core as BERTrend Core
    participant Embedding as Embedding Service
    participant TopicModel as BERTopic Model
    participant SignalAnalysis as Signal Analysis

    User->>App: Load data
    App->>Core: Process data
    Core->>Embedding: Embed documents
    Embedding->>Core: Return embeddings
    Core->>TopicModel: Train topic models
    TopicModel->>Core: Return topic models
    Core->>SignalAnalysis: Analyze signals
    SignalAnalysis->>Core: Return signal classifications
    Core->>App: Return results
    App->>User: Display results
```

## Data Flow

```mermaid
flowchart TD
    RawData[Raw Text Data] --> Preprocessing[Preprocessing]
    Preprocessing --> Embedding[Document Embedding]
    Embedding --> TopicModeling[Topic Modeling]
    TopicModeling --> TopicMerging[Topic Merging]
    TopicMerging --> SignalAnalysis[Signal Analysis]
    SignalAnalysis --> Classification[Signal Classification]
    Classification --> Visualization[Visualization]

    subgraph DataTypes[Data Types]
        RawData --> |Text Documents| Preprocessing
        Preprocessing --> |Cleaned Text| Embedding
        Embedding --> |Document Vectors| TopicModeling
        TopicModeling --> |Topic Models| TopicMerging
        TopicMerging --> |Merged Topics| SignalAnalysis
        SignalAnalysis --> |Signal Trends| Classification
        Classification --> |Classified Signals| Visualization
    end
```

## Class Diagram

```mermaid
classDiagram
    class BERTrend {
        +train_topic_models()
        +merge_models_with()
        +calculate_signal_popularity()
        +classify_signals()
        +save_model()
        +restore_model()
    }

    class BERTopicModel {
        +fit()
        +get_default_config()
    }

    class EmbeddingService {
        +embed()
        -_local_embed_documents()
        -_remote_embed_documents()
    }

    BERTrend --> BERTopicModel : uses
    BERTrend --> EmbeddingService : uses
    BERTopicModel --> "BERTopic Library" : wraps
    EmbeddingService --> "SentenceTransformer" : uses
```

## Process Diagram: Topic Model Training

```mermaid
stateDiagram-v2
    [*] --> LoadData
    LoadData --> Preprocess
    Preprocess --> EmbedDocuments
    EmbedDocuments --> TrainTopicModels
    TrainTopicModels --> MergeModels
    MergeModels --> CalculateSignalPopularity
    CalculateSignalPopularity --> ClassifySignals
    ClassifySignals --> SaveResults
    SaveResults --> [*]
```

## Process Diagram: Signal Classification

```mermaid
stateDiagram-v2
    [*] --> CalculatePopularity
    CalculatePopularity --> ComputeThresholds
    ComputeThresholds --> ClassifyTopics

    state ClassifyTopics {
        [*] --> CheckPopularity
        CheckPopularity --> Noise : Popularity < q1
        CheckPopularity --> CheckTrend : q1 <= Popularity <= q3
        CheckPopularity --> StrongSignal : Popularity > q3

        CheckTrend --> WeakSignal : Rising
        CheckTrend --> Noise : Not Rising
    }

    ClassifyTopics --> SaveClassification
    SaveClassification --> [*]
```

## Deployment Architecture

```mermaid
flowchart TD
    subgraph User[User Environment]
        Browser[Web Browser]
        CLI[Command Line Interface]
    end

    subgraph AppServer[Application Server]
        StreamlitApp[Streamlit Application]
        BERTrendLib[BERTrend Library]
    end

    subgraph ComputeResources[Compute Resources]
        CPU[CPU Processing]
        GPU[GPU Acceleration]
    end

    subgraph ExternalServices[External Services]
        EmbeddingAPI[Embedding API]
        LLMAPI[LLM API]
    end

    subgraph Storage[Data Storage]
        FileSystem[File System]
        ModelCache[Model Cache]
    end

    Browser --> StreamlitApp
    CLI --> BERTrendLib
    StreamlitApp --> BERTrendLib
    BERTrendLib --> CPU
    BERTrendLib --> GPU
    BERTrendLib --> EmbeddingAPI
    BERTrendLib --> LLMAPI
    BERTrendLib --> FileSystem
    BERTrendLib --> ModelCache
```

## Module Dependencies

```mermaid
graph TD
    subgraph BERTrendApps[BERTrend Apps]
        Prospective[Prospective Demo]
        Exploration[Exploration Tools]
        Newsletters[Newsletter Analysis]
    end

    subgraph BERTrendCore[BERTrend Core]
        BERTrendClass[BERTrend Class]
        TopicAnalysis[Topic Analysis]
        TrendAnalysis[Trend Analysis]
        LLMUtils[LLM Utilities]
    end

    subgraph Utils[Utilities]
        DataLoading[Data Loading]
        Caching[Caching]
        Metrics[Metrics]
    end

    Prospective --> BERTrendClass
    Exploration --> BERTrendClass
    Newsletters --> BERTrendClass

    BERTrendClass --> TopicAnalysis
    BERTrendClass --> TrendAnalysis
    BERTrendClass --> LLMUtils

    BERTrendClass --> DataLoading
    BERTrendClass --> Caching
    TopicAnalysis --> Metrics
```
