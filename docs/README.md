# BERTrend Documentation

Index of the documentation available in this folder.

## Architecture and data structures

- **<a href="data_architecture.html">Data architecture and functions</a>** — *interactive page*: click-to-expand view of the
  files produced by BERTrend (`bertrend.dill` and the other artefacts), the BERTrend functions, and the relational
  schema. GitHub does not render HTML files, so open it from the documentation site, or download it and open it in a
  browser.
- [BERTrend model structure](bertrend_model_structure.md) — internal attributes of a serialized `BERTrend` model.
- [System architecture](services/architecture.md) — components and their interactions.
- [Data flows](flows.md) — end-to-end processing flows.
- [Code dependencies](code_dependencies.md)
- [Concurrency](concurrency.md)

## Services

- [Services architecture](services/architecture.md)
- [BERTrend API service](services/bertrend_api_service.md)
- [Embedding service](services/embedding_service.md)
- [Article scoring service](services/article_scoring_service.md)
- [Summary service](services/summary_service.md)
- [Scheduling service](services/scheduling_service.md) and [scheduler configuration](services/scheduler_configuration.md)
- [Queue architecture](services/queue_architecture.md)

## Demos

- [Overview of the demos](demos/demos.md)
- [Topic analysis demo](demos/topic_analysis_demo.md)
- [Weak signals demo](demos/weak_signals_demo.md)
- [Prospective demo](demos/prospective_demo.md) and its [architecture](demos/prospective_demo_architecture.md)

## Usage guides

- [Running BERTrend with Docker](docker.md)
- [Data providers](data_provider.md)
- [Mail configuration](mail_configuration.md)
- [Newsletters](newsletters.md)
- [Automated report generation](automated_report_generation.md)
- [Article scoring](article_scoring.md)
- [Outputs of `process_new_data`](process_new_data_outputs.md)

## API reference

- [API documentation index](api/README.md)
