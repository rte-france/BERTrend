# Metrics API Reference

## Module: `bertrend/metrics/`

This module provides temporal topic stability metrics (TEMPTopic) and standard topic quality metrics.

---

## TEMPTopic (`temporal_metrics_embedding.py`)

The `TempTopic` class extends BERTopic evaluation with temporal stability metrics using embeddings. It measures how
consistently topics are represented across time periods.

### Class: `TempTopic`

```python
TempTopic(
    topic_model: BERTopic,
    topics_over_time: pd.DataFrame,
    evolution_tuning: bool = True,
    global_tuning: bool = True,
    embedding_model: SentenceTransformer = None,
)
```

### Methods

#### `fit()`

Compute topics-over-time using BERTopic's internal method. Must be called before stability calculations.

#### `calculate_temporal_representation_stability()`

Calculate how stable topic **word representations** are across consecutive time windows. Returns a DataFrame with
per-topic stability scores.

#### `calculate_topic_embedding_stability()`

Calculate how stable topic **embeddings** are across consecutive time windows. Uses cosine similarity between topic
embedding vectors at adjacent timestamps.

#### `calculate_overall_topic_stability()`

Aggregate representation and embedding stability into a single overall stability score per topic.

#### `find_similar_topic_pairs()`

Identify pairs of topics that are highly similar (potential duplicates or closely related topics).

### Usage Example

```python
from bertrend.metrics.temporal_metrics_embedding import TempTopic

temp_topic = TempTopic(
    topic_model=fitted_bertopic_model,
    topics_over_time=topics_over_time_df,
    embedding_model=sentence_transformer,
)
temp_topic.fit()

repr_stability = temp_topic.calculate_temporal_representation_stability()
emb_stability = temp_topic.calculate_topic_embedding_stability()
overall = temp_topic.calculate_overall_topic_stability()
```

---

## Topic Quality Metrics (`topic_metrics.py`)

Standalone functions for evaluating topic model quality.

### `get_coherence_value(topic_model, docs)`

Compute topic coherence (C_v) for a fitted BERTopic model.

| Parameter     | Type        | Description                          |
|---------------|-------------|--------------------------------------|
| `topic_model` | `BERTopic`  | Fitted topic model.                  |
| `docs`        | `list[str]` | Original documents used for fitting. |

**Returns:** `float` — coherence score.

### `get_diversity_value(topic_model)`

Compute topic diversity using pairwise Jaccard distance between topic word sets.

**Returns:** `float` — diversity score (0–1, higher = more diverse).

### `compute_cluster_metrics(topic_model, embeddings)`

Compute clustering quality metrics (e.g., silhouette score) on the topic assignments.

### `proportion_unique_words(topics)`

Calculate the proportion of unique words across all topics.

### `pairwise_jaccard_diversity(topics)`

Calculate pairwise Jaccard diversity across topic word lists.
