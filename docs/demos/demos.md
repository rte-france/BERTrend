# BERTrend Demos

This document provides an overview of the available demos in the BERTrend project. Each demo showcases different
capabilities of the BERTrend framework for topic modeling, trend analysis, and signal detection.

## Available Demos

BERTrend offers three main demo applications:

1. [Topic Analysis Demo](#topic-analysis-demo)
2. [Weak Signals Demo](#weak-signals-demo)
3. [Prospective Demo](#prospective-demo)

### Topic Analysis Demo

The Topic Analysis Demo provides a Streamlit-based web application for topic analysis using BERTopic. This application
allows users to load data, train topic models, explore topics, visualize topic distributions and relationships, analyze
temporal trends, and generate newsletters based on the analysis.

**Key Features:**

- Data loading and model training
- Topic exploration and visualization
- Temporal analysis of topic evolution
- Newsletter generation based on topic analysis

[Learn more about the Topic Analysis Demo](topic_analysis_demo.md)

### Weak Signals Demo

The Weak Signals Demo offers a comprehensive web application for detecting and analyzing weak signals in topic models
over time using BERTrend's topic modeling capabilities. This application allows users to load data, train models, and
analyze the evolution of topics and signals over time.

**Key Features:**

- Data loading and embedding
- Model training for specific time periods
- Signal analysis (noise, weak signals, strong signals)
- Topic evolution visualization
- State management for saving and restoring application state

[Learn more about the Weak Signals Demo](weak_signals_demo.md)

### Prospective Demo

The Prospective Demo provides a comprehensive web application for prospective analysis using BERTrend's topic modeling
and signal detection capabilities. This application allows users to monitor information sources, analyze trends and
signals, and generate reports based on the analysis.

**Key Features:**

- Information source management
- Model management
- Signal analysis
- Dashboard analysis
- Report generation

[Learn more about the Prospective Demo](prospective_demo.md)

## Getting Started

### Environment (.env)

BERTrend supports configuration via a repository-level .env file that is auto-loaded on import when python-dotenv is
installed. Create and fill .env at the repo root (a template exists) and set values like:

- BERTREND_BASE_DIR: base directory for BERTrend data/models/logs
- OpenAI/LLM: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL
- Optional providers: SEMANTIC_SCHOLAR_API_KEY, NEWSCATCHER_API_KEY, DBPEDIA_REST_API_URL
- Optional email, embedding server security, and CUDA_VISIBLE_DEVICES

If python-dotenv is not installed, export these variables via your shell before starting the demos.

To run any of the demos, navigate to the respective directory and run the Streamlit application:

### Topic Analysis Demo

```bash
cd bertrend/demos/topic_analysis
streamlit run app.py
```

### Weak Signals Demo

```bash
cd bertrend/demos/weak_signals
streamlit run app.py
```

### Prospective Demo

```bash
cd bertrend_apps/prospective_demo
CUDA_VISIBLE_DEVICES=<gpu_number> streamlit run app.py
```

## Dependencies

All demos depend on:

- BERTrend core functionality
- BERTopic for topic modeling
- Streamlit for the web interface
- Various data processing and visualization libraries

These dependencies are included in the main package requirements or in the "apps" optional dependency group.