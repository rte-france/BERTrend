# BERTrend Weak Signals Demo

## Overview

The `bertrend.demos.weak_signals` package provides a comprehensive web application for detecting and analyzing weak signals in topic models over time using BERTrend's topic modeling capabilities. This application allows users to load data, train models, and analyze the evolution of topics and signals over time.

## Features

The Weak Signals Demo application offers the following key features:

1. **Data Loading and Embedding**
   - Load and preprocess textual data
   - Embed documents using configurable embedding models

2. **Model Training**
   - Train BERTopic models for specific time periods
   - Merge models to track topic evolution over time

3. **Signal Analysis**
   - Identify and categorize signals as noise, weak signals, or strong signals
   - Analyze signal evolution over time
   - Perform detailed analysis of individual signals

4. **Topic Evolution Visualization**
   - Visualize topic evolution using Sankey diagrams
   - Track newly emerged topics
   - Monitor topic popularity evolution

5. **State Management**
   - Save and restore application state
   - Cache management for improved performance

## Components

The package consists of several key components:

### Main Application

The main application (`app.py`) provides a Streamlit-based web interface with multiple tabs:
- "Data Loading" - for loading and embedding textual data
- "Model Training" - for training and merging topic models
- "Results Analysis" - for analyzing signals and topic evolution

### Visualization Utilities

The package includes extensive visualization utilities (`visualizations_utils.py`):
- Sankey diagrams for topic evolution
- Signal categorization displays
- Topic popularity evolution charts
- Signal analysis visualizations
- Topic count retrieval and display

### User Messages

The package includes a comprehensive set of user messages (`messages.py`) for:
- Success notifications
- Error warnings
- Progress indicators

## Environment (.env)

BERTrend auto-loads a repository-level .env on import when python-dotenv is installed. Before running the Weak Signals Demo, set relevant variables in the repo .env, for example:
- BERTREND_BASE_DIR: base directory for BERTrend data/models/logs
- OpenAI/LLM: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL
- Optional providers and CUDA_VISIBLE_DEVICES as needed

If python-dotenv isn’t installed, export these variables via your shell.

## Usage

### Starting the Application

To start the Weak Signals Demo application:

```bash
cd bertrend/demos/weak_signals
streamlit run app.py
```

### Data Loading and Embedding

1. In the "Data Loading" tab:
   - Load your textual data
   - Configure embedding parameters
   - Embed the documents

### Model Training

1. In the "Model Training" tab:
   - Configure BERTopic hyperparameters
   - Train models for specific time periods
   - Merge models to track topic evolution

### Signal Analysis

1. In the "Results Analysis" tab:
   - View signal categorization (noise, weak signals, strong signals)
   - Analyze topic evolution using Sankey diagrams
   - Examine newly emerged topics
   - Track topic popularity evolution
   - Perform detailed analysis of individual signals

### State Management

The application provides state management capabilities:
- Save the current application state
- Restore a previous application state
- Purge cache to free up resources

## Configuration

The application provides configuration options for:
- Embedding hyperparameters
- BERTopic hyperparameters
- BERTrend hyperparameters

These can be configured through the sidebar in the application.

## Working with large datasets

The **Data Loading** tab offers two ways to provide data:

- **Local data** uses Streamlit's file uploader, which is capped by Streamlit's
  `maxUploadSize` (**200 MB by default**). Large corpora (e.g. an arXiv dump)
  will exceed this limit.
- **Remote data** lists the compatible files found in the BERTrend data
  directory (`DATA_PATH`, i.e. `$BERTREND_BASE_DIR/data`, default
  `~/.bertrend/data`) and has **no size limit**.

To use a large dataset with the demo, prefer one of:

1. **Place the file in `DATA_PATH`** and select it from the **Remote data**
   tab. This bypasses the uploader size limit entirely (recommended).
2. **Raise the uploader limit** by creating a `.streamlit/config.toml` next to
   the app with:
   ```toml
   [server]
   maxUploadSize = 4000  # in MB
   ```
   or by launching with `streamlit run app.py --server.maxUploadSize 4000`.

Supported input formats are `.csv`, `.parquet`, and `.jsonl(.gz)`. Each file
must contain at least a **`text`** column and a **`timestamp`** column; the
optional `url`, `title`, `source`, and `document_id` columns are used when
present.

## Running BERTrend without the demo (programmatic API)

For large datasets, batch runs, or reproducing published results, it is often
easier to drive BERTrend directly from Python rather than through the Streamlit
app. The notebooks in [`getting_started/`](../../getting_started) show the full
retrospective-analysis pipeline end to end:

- [`bertrend_quickstart.ipynb`](../../getting_started/bertrend_quickstart.ipynb)
- [`explore_bertrend_model.ipynb`](../../getting_started/explore_bertrend_model.ipynb)

The core steps are:

```python
from pathlib import Path
from bertrend.BERTrend import BERTrend
from bertrend.BERTopicModel import BERTopicModel
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.data_loading import load_data, group_by_days
from bertrend.trend_analysis.weak_signals import analyze_signal

# 1. Configure the topic model and BERTrend
topic_model = BERTopicModel({"global": {"language": "English"}})
bertrend = BERTrend(topic_model=topic_model)

# 2. Load a DataFrame with (at least) `text` and `timestamp` columns
df = load_data(Path("my_dataset.jsonl"), language="English")

# 3. Embed the documents (local model or remote embedding server)
embedding_service = EmbeddingService(local=True)
embeddings, _, _ = embedding_service.embed(texts=df["text"])

# 4. Split into time slices
grouped_data = group_by_days(df=df, day_granularity=30)

# 5. Train the per-period topic models and merge them over time
bertrend.train_topic_models(
    grouped_data=grouped_data,
    embedding_model=embedding_service.embedding_model_name,
    embeddings=embeddings,
)

# 6. Compute popularity and classify signals over time
bertrend.calculate_signal_popularity()
for ts in bertrend.doc_groups.keys():
    noise_df, weak_df, strong_df = bertrend.classify_signals(
        window_size=30, current_date=ts
    )

# 7. (Optional) LLM-based interpretation of a given signal/topic
summary, analysis = analyze_signal(bertrend, topic_number=1, current_date=ts)
```

The plotting helpers used by the demo live in
`bertrend/demos/weak_signals/visualizations_utils.py` and
`bertrend/trend_analysis/weak_signals.py`, and can be reused on the objects
produced above (signal classification and popularity evolution) to build the
figures.

> **Note:** the repository ships the BERTrend *method* and the notebooks above,
> but not a one-click script for a specific paper figure nor a frozen copy of
> the arXiv corpus used in the paper. An arXiv data provider is available
> (`bertrend/bertrend_apps/data_provider/arxiv_provider.py`, with
> `bertrend/bertrend_apps/config/feeds/arxiv_feed.toml`) to fetch data, but
> exact reproduction requires matching the original query, date range,
> `granularity`, and `window_size`.

## Dependencies

The Weak Signals Demo package depends on:
- BERTrend core functionality
- BERTopic for topic modeling
- Streamlit for the web interface
- Plotly for interactive visualizations
- Pandas for data manipulation