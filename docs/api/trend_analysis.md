# Trend Analysis API Reference

## Module: `bertrend/trend_analysis/`

Functions for analyzing topic trends over time, detecting weak signals, and visualizing trend evolution.

---

## Weak Signals Detection (`weak_signals.py`)

Core logic for identifying and tracking weak signals in topic evolution data.

### `detect_weak_signals_zeroshot(bertrend, window_size, ...)`

Detect weak signals using zero-shot topic classification combined with popularity-based filtering.

### `analyze_signal(signal_df, ...)`

Analyze a single signal's trajectory and classify it (emerging, growing, stable, declining).

### Internal Helpers

| Function                                                   | Description                                                      |
|------------------------------------------------------------|------------------------------------------------------------------|
| `_apply_decay(popularity_dict, decay_factor, decay_power)` | Apply exponential decay to topic popularity scores.              |
| `_filter_data(df, window_start, window_end)`               | Filter a DataFrame to a specific time window.                    |
| `_is_rising_popularity(topic_popularity)`                  | Determine if a topic's popularity is increasing.                 |
| `_create_df(topics, ...)`                                  | Create a DataFrame from topic data for a single signal category. |
| `_create_dataframes(weak, strong, noise)`                  | Create DataFrames for all signal categories.                     |
| `_initialize_new_topic(topic_id, ...)`                     | Initialize tracking state for a newly detected topic.            |
| `update_existing_topic(topic_id, ...)`                     | Update tracking state for an existing topic with new data.       |
| `_apply_decay_to_inactive_topics(...)`                     | Apply decay to topics not seen in the current period.            |

---

## Data Structures (`data_structure.py`)

Pydantic models for structured trend analysis outputs (used with LLM-based interpretation).

| Class                   | Description                                                   |
|-------------------------|---------------------------------------------------------------|
| `TopicSummary`          | Summary of a single topic: title, description, key documents. |
| `TopicSummaryList`      | List container for multiple `TopicSummary` items.             |
| `PotentialImplications` | LLM-generated implications of a trend.                        |
| `EvolutionScenario`     | Possible future evolution scenarios for a signal.             |
| `TopicInterconnexions`  | Relationships and connections between topics.                 |
| `Drivers`               | Identified drivers behind a trend.                            |
| `SignalAnalysis`        | Complete analysis of a signal combining all above components. |

---

## Visualizations (`visualizations.py`)

Plotly-based visualization functions for trend analysis.

### `plot_num_topics(merge_df, ...)`

Plot the number of active topics over time.

### `plot_size_outliers(merge_df, ...)`

Plot outlier topics by size.

### `plot_topics_for_model(topic_model, timestamp, ...)`

Visualize topics for a single time-period model.

### `create_topic_size_evolution_figure(merge_df, ...)` / `plot_topic_size_evolution(...)`

Create and display a figure showing how topic sizes evolve across periods.

### `plot_newly_emerged_topics(merge_df, ...)`

Highlight topics that appeared for the first time in each period.

### `create_sankey_diagram_plotly(merge_df, ...)`

Generate a Sankey diagram showing topic merge/split flows across time periods.

### `find_connected_nodes(merge_df, topic_id)`

Trace all connected topics (predecessors and successors) for a given topic through the merge history.

---

## Prompts (`prompts.py`)

Prompt templates for LLM-based trend interpretation, including:

- Signal analysis and implication generation
- Evolution scenario generation
- Topic interconnection analysis
