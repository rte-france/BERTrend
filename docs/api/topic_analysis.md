# Topic Analysis API Reference

## Module: `bertrend/topic_analysis/`

Functions for describing, extracting, and visualizing topics from fitted BERTopic models.

---

## Topic Description (`topic_description.py`)

### `get_topic_description(topic_model, topic_id)`

Retrieve the keyword-based description for a specific topic from a fitted model.

| Parameter     | Type       | Description            |
|---------------|------------|------------------------|
| `topic_model` | `BERTopic` | Fitted BERTopic model. |
| `topic_id`    | `int`      | Topic identifier.      |

**Returns:** `str` — Comma-separated top keywords for the topic.

### `generate_topic_description(topic_model, topic_id, docs, ...)`

Generate a natural-language topic description using an LLM, based on the topic's keywords and representative documents.

---

## Representative Documents (`representative_docs.py`)

### `get_most_representative_docs(topic_model, topic_id, n)`

Extract the most representative documents for a given topic.

| Parameter     | Type       | Description                    |
|---------------|------------|--------------------------------|
| `topic_model` | `BERTopic` | Fitted BERTopic model.         |
| `topic_id`    | `int`      | Topic identifier.              |
| `n`           | `int`      | Number of documents to return. |

**Returns:** `list[str]` — Most representative document texts.

---

## Data Structures (`data_structure.py`)

### `TopicDescription`

Pydantic model representing a structured topic description with fields for keywords, summary, and metadata.

---

## Visualizations (`visualizations.py`)

Plotly-based visualization functions for topic analysis results.

### `plot_topics_over_time(topics_over_time_df, ...)`

Plot topic prevalence over time as a line chart.

### `plot_docs_repartition_over_time(df, ...)`

Plot the distribution of documents across topics over time.

### `plot_remaining_docs_repartition_over_time(df, ...)`

Plot the distribution of documents not assigned to major topics.

### `plot_topic_evolution(topic_model, topics_over_time, topic_id, ...)`

Plot the evolution of a single topic's representation and size over time.

### `plot_temporal_stability_metrics(stability_df, ...)`

Visualize TEMPTopic temporal stability metrics (representation and embedding stability) per topic.

### `plot_overall_topic_stability(overall_stability_df, ...)`

Bar chart of overall topic stability scores.

---

## Prompts (`prompts.py`)

Prompt templates used by the LLM for generating topic descriptions and summaries within the topic analysis workflow.
