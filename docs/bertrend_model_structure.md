### BERTrend Model Structure

A BERTrend model is a comprehensive state object that tracks the evolution of topics over time. It extends the capabilities of BERTopic by maintaining a longitudinal view of topic emergence, merging, and popularity.

#### 1. Serialization Overview

BERTrend models are typically saved using the `dill` library, which allows for serializing complex Python objects, including nested dictionaries and custom classes. By default, models are saved in the directory specified by `MODELS_DIR` (often `~/.bertrend/cache/models`).

#### 2. Core Attributes

The `BERTrend` class contains several key attributes that store the analysis results:

| Attribute | Type | Description |
|-----------|------|-------------|
| `last_topic_model` | `bertopic.BERTopic` | The underlying BERTopic model trained on the most recent time period. |
| `merged_df` | `pd.DataFrame` | The primary data structure containing information about all topics tracked by the model across all periods. |
| `all_merge_histories_df` | `pd.DataFrame` | A log of all merge operations between topics from consecutive time periods. |
| `all_new_topics_df` | `pd.DataFrame` | A record of newly emerged topics at each time period. |
| `topic_sizes` | `dict` | A dictionary tracking the popularity (size) and document counts of each topic over time. |
| `doc_groups` | `dict` | Maps each timestamp to the list of documents processed in that period. |
| `emb_groups` | `dict` | Maps each timestamp to the document embeddings for that period. |

#### 3. Detailed Data Structures

##### `merged_df` (DataFrame)
This is the central repository of topic information. Each row represents a "global" topic tracked by BERTrend.

- **Topic**: Unique ID for the merged topic.
- **Count**: Total number of paragraphs/segments associated with the topic.
- **Document_Count**: Total number of unique source documents.
- **Representation**: Key terms representing the topic (from the latest period).
- **Documents**: A list of tuples `(timestamp, documents_list)` tracking the documents added to this topic over time.
- **Embedding**: Centroid embedding of the topic.
- **Sources**: List of data sources (e.g., RSS feed names) contributing to this topic.
- **URLs**: List of URLs for the source documents.

##### `all_merge_histories_df` (DataFrame)
Tracks how topics from a new period (Topic2) were merged into existing topics (Topic1).

- **Timestamp**: When the merge occurred.
- **Topic1**: ID of the existing topic in the model.
- **Topic2**: ID of the topic in the new period's model.
- **Similarity**: Cosine similarity between the two topics.
- **Representation1/2**: Term representations for both topics at the time of merge.

##### `topic_sizes` (Dictionary)
Used for popularity trend analysis and weak signal detection.
Structure: `{topic_id: {"Timestamps": [...], "Popularity": [...], "Docs_Count": [...]}}`

- **Popularity**: Normalized size of the topic in a given period (after applying decay if configured).

#### 4. Usage in Analysis

- **Trend Visualization**: `merged_df` and `topic_sizes` are used to plot the evolution of topic popularity over time.
- **Sankey Diagrams**: `all_merge_histories_df` is used to visualize the flow and merging of topics between periods.
- **Weak Signal Detection**: BERTrend uses `topic_sizes` to identify topics with low but rising popularity, classifying them as weak signals based on configurable thresholds (Q1/Q3).
- **Signal Implications**: The `analyze_signal` function uses the representative documents and topic history from these structures to generate LLM-based insights.
