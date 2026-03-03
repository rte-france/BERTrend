# Core Library API Reference

## BERTrend (`bertrend/BERTrend.py`)

The `BERTrend` class is the main entry point for trend analysis and weak signal detection using BERTopic.

### Overview

`BERTrend` trains BERTopic models on time-grouped document corpora, merges topic models across time periods, computes
signal popularity with configurable decay, and classifies signals into categories (noise, weak, strong).

### Constructor

```python
BERTrend(
    config_file: str | Path = BERTREND_DEFAULT_CONFIG_PATH,
    topic_model: BERTopicModel = None,
)
```

| Parameter     | Type            | Description                                                                                |
|---------------|-----------------|--------------------------------------------------------------------------------------------|
| `config_file` | `str \| Path`   | Path to a TOML configuration file. Defaults to `bertrend/config/bertrend_config.toml`.     |
| `topic_model` | `BERTopicModel` | Optional pre-configured BERTopicModel instance. If `None`, one is created from the config. |

### Key Instance Attributes

| Attribute                    | Type                           | Description                                     |
|------------------------------|--------------------------------|-------------------------------------------------|
| `topic_models`               | `dict[pd.Timestamp, BERTopic]` | Trained topic models keyed by time period.      |
| `merge_df`                   | `pd.DataFrame`                 | DataFrame tracking topic merges across periods. |
| `doc_groups`                 | `dict`                         | Document groups per timestamp.                  |
| `emb_groups`                 | `dict`                         | Embedding groups per timestamp.                 |
| `last_topic_model`           | `BERTopic`                     | Most recently trained/merged topic model.       |
| `last_topic_model_timestamp` | `pd.Timestamp`                 | Timestamp of the last model.                    |
| `signal_popularity`          | `dict`                         | Computed signal popularity scores.              |

### Methods

#### `train_topic_models(grouped_data, embedding_model, embeddings, bertrend_models_path, save_topic_models)`

Train BERTopic models for each timestamp in the grouped data and merge them sequentially.

| Parameter              | Type                               | Default      | Description                        |
|------------------------|------------------------------------|--------------|------------------------------------|
| `grouped_data`         | `dict[pd.Timestamp, pd.DataFrame]` | —            | Documents grouped by timestamp.    |
| `embedding_model`      | `SentenceTransformer \| str`       | —            | Embedding model or model name.     |
| `embeddings`           | `np.ndarray`                       | —            | Precomputed document embeddings.   |
| `bertrend_models_path` | `Path`                             | `MODELS_DIR` | Directory to save models.          |
| `save_topic_models`    | `bool`                             | `True`       | Whether to persist models to disk. |

Updates instance variables: `doc_groups`, `emb_groups`, `last_topic_model`, `last_topic_model_timestamp`.

#### `merge_models_with(new_model, new_model_timestamp, min_similarity)`

Merge a new BERTopic model with the last trained model, tracking topic evolution in `merge_df`.

| Parameter             | Type           | Default | Description                                                         |
|-----------------------|----------------|---------|---------------------------------------------------------------------|
| `new_model`           | `BERTopic`     | —       | New topic model to merge.                                           |
| `new_model_timestamp` | `pd.Timestamp` | —       | Timestamp of the new model.                                         |
| `min_similarity`      | `int \| None`  | `None`  | Minimum cosine similarity for merging. Uses config value if `None`. |

#### `calculate_signal_popularity(decay_factor, decay_power)`

Compute popularity scores for all topics across time, applying exponential decay to older signals.

| Parameter      | Type            | Default | Description                                           |
|----------------|-----------------|---------|-------------------------------------------------------|
| `decay_factor` | `float \| None` | `None`  | Decay factor for signal aging. Uses config if `None`. |
| `decay_power`  | `float \| None` | `None`  | Decay power exponent. Uses config if `None`.          |

#### `classify_signals(window_size, current_date)`

Classify topics into **noise**, **weak signals**, and **strong signals** based on popularity quartiles within a time
window.

| Parameter      | Type           | Description                            |
|----------------|----------------|----------------------------------------|
| `window_size`  | `int`          | Number of periods to look back.        |
| `current_date` | `pd.Timestamp` | End date of the classification window. |

**Returns:** Updates internal signal classification state. Use `save_signal_evolution_data()` to export.

#### `save_model(models_path)` / `restore_model(models_path)` (classmethod)

Serialize/deserialize the full `BERTrend` state (merge history, popularity data, etc.) using `dill`.

#### `save_topic_model(period, topic_model, models_path)` / `restore_topic_model(period, models_path)` (classmethods)

Save/load individual BERTopic models for a specific time period.

#### `restore_topic_models(models_path)`

Restore all previously saved topic models into `self.topic_models`.

#### `save_signal_evolution_data(window_size, start_timestamp, end_timestamp)`

Export signal classification results (weak signals, strong signals, noise) to Parquet files.

### Module-Level Functions

#### `train_new_data(reference_timestamp, new_data, bertrend_models_path, embedding_service, granularity, language)`

Train a new topic model on incoming data and merge it with an existing `BERTrend` model. Used by the worker/service
layer for incremental updates.

#### `_preprocess_model(topic_model, docs, embeddings)`

Internal helper that filters outlier topics and prepares DataFrames for merging.

#### `_merge_models(df1, df2, min_similarity, timestamp)`

Internal helper that performs the actual cosine-similarity-based topic merge between two DataFrames.

---

## BERTopicModel (`bertrend/BERTopicModel.py`)

A configuration and wrapper layer around [BERTopic](https://maartengr.github.io/BERTopic/) that manages sub-model
initialization and representation strategies.

### Classes

#### `BERTopicModelOutput`

Container for a fitted BERTopic model and its outputs.

```python
class BERTopicModelOutput:
    def __init__(self, topic_model: BERTopic)
```

Attributes: `topic_model`, `topics`, `probs`, `topic_info`, `doc_info`.

#### `BERTopicModel`

```python
class BERTopicModel:
    def __init__(self, config: str | Path | dict = BERTOPIC_DEFAULT_CONFIG_PATH)
```

| Parameter | Type                  | Description                                         |
|-----------|-----------------------|-----------------------------------------------------|
| `config`  | `str \| Path \| dict` | Path to a TOML config file or a dict of parameters. |

**Configuration keys** (from TOML):

| Key                       | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| `language`                | Language for stopwords and processing.                             |
| `min_topic_size`          | Minimum documents per topic.                                       |
| `top_n_words`             | Number of top words per topic.                                     |
| `n_gram_range`            | N-gram range for vectorizer.                                       |
| `zeroshot_topic_list`     | Optional list of seed topics for zero-shot classification.         |
| `zeroshot_min_similarity` | Minimum similarity for zero-shot matching.                         |
| `representation_model`    | List of representation strategies (e.g., `["KeyBERT", "OpenAI"]`). |

### Methods

#### `get_default_config()` (classmethod)

Returns the default configuration dictionary loaded from the bundled TOML file.

#### `fit(docs, embeddings, embedding_model, zeroshot_topic_list, zeroshot_min_similarity)`

Fit a BERTopic model on the provided documents and embeddings.

| Parameter                 | Type                                 | Default | Description                               |
|---------------------------|--------------------------------------|---------|-------------------------------------------|
| `docs`                    | `list[str]`                          | —       | Documents to model.                       |
| `embeddings`              | `np.ndarray`                         | —       | Precomputed embeddings.                   |
| `embedding_model`         | `SentenceTransformer \| str \| None` | `None`  | Embedding model (for BERTopic internals). |
| `zeroshot_topic_list`     | `list[str] \| None`                  | `None`  | Seed topics for zero-shot mode.           |
| `zeroshot_min_similarity` | `float \| None`                      | `None`  | Similarity threshold for zero-shot.       |

**Returns:** `BERTopicModelOutput`
