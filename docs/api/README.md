# BERTrend API Documentation

This directory contains detailed API reference documentation for all major BERTrend modules.

## Core Library

- **[Core Library API](core_library.md)** — `BERTrend` class (model training, merging, signal classification) and
  `BERTopicModel` wrapper.
- **[Metrics](metrics.md)** — TEMPTopic temporal stability metrics and topic quality metrics.

## Analysis

- **[Topic Analysis](topic_analysis.md)** — Topic description, representative documents, and visualizations.
- **[Trend Analysis](trend_analysis.md)** — Weak signal detection, trend evolution, Sankey diagrams, and data
  structures.

## LLM & Scoring

- **[LLM Utilities](llm_utils.md)** — OpenAI client, agent utilities, newsletter generation, and prompt templates.
- **[Article Scoring](article_scoring.md)** — LLM-powered article quality scoring system.

## Services & Infrastructure

- **[Services](services.md)** — FastAPI routers, request/response models, queue worker, queue monitor, and logging
  utilities.

## Data Providers

- **[Data Providers](data_providers.md)** — RSS, Atom, ArXiv, Google News, Bing News, Newscatcher, and Deep Research
  providers.
