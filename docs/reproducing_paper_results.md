# Reproducing the paper results on arXiv

This guide explains how to run BERTrend on the arXiv dataset used in the
[BERTrend paper](https://github.com/rte-france/BERTrend/blob/main/BERTrend%20paper.pdf),
without the Streamlit demo, and how to build the data behind:

- **Figure 2(b)**: evolution of signal types and topic counts on arXiv cs.* papers;
- **Figure 3(b)**: log-scaled popularity of selected arXiv topics (attention models,
  ImageNet, RNN/LSTM).

The demo is convenient for exploration, but for a corpus of this size (several
hundred thousand abstracts) the programmatic API is easier to run and to
reproduce. See also [Running BERTrend without the demo](demos/weak_signals_demo.md#running-bertrend-without-the-demo-programmatic-api).

> **Note:** results will be close to, but not identical to, the published
> figures: topic identifiers, exact counts and curves depend on the arXiv
> snapshot date, library versions (BERTopic, UMAP, HDBSCAN) and hardware.
> The qualitative behaviour (signal type distributions, rise of the attention
> topic after June 2017) is what should be reproduced.

---

## 1. Get the arXiv dataset

The paper uses the arXiv metadata published by Cornell University, available on
Kaggle as [`Cornell-University/arxiv`](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(file `arxiv-metadata-oai-snapshot.json`, one JSON record per line).

The paper keeps the **computer science** papers (`cs.*` categories) from
**January 2017 to December 2023**, i.e. **367,248 abstracts**. Your count may
differ slightly depending on the snapshot date and on whether cross-listed papers
are kept (`PRIMARY_ONLY` below); use it as a sanity check.

## 2. Parameters used in the paper

| Parameter                         | Paper (arXiv)                   | Where it is set                                   | Library default |
|-----------------------------------|---------------------------------|---------------------------------------------------|-----------------|
| Embedding model                   | `all-mpnet-base-v2`             | `EmbeddingService(model_name=...)`                | service config  |
| UMAP `n_neighbors` / `n_components` | 15 / 5                        | `BERTopicModel` → `[umap_model]`                  | 5 / 5           |
| HDBSCAN `min_cluster_size` / `min_samples` | 2 / 1                  | `BERTopicModel` → `[hdbscan_model]`               | 5 / 5           |
| Topic representation              | unigrams + bigrams (c-TF-IDF)   | `BERTopicModel` → `[vectorizer_model] ngram_range` | `[1, 2]`       |
| Time granularity *G*              | 7 days                          | `BERTrend` config `granularity` + `group_by_days` | 2               |
| Merge similarity threshold α      | 0.7                             | `BERTrend` config `min_similarity`                | 0.7             |
| Decay factor λ                    | 0.01                            | `BERTrend` config `decay_factor`                  | 0.01            |
| Classification percentiles        | P10 / P50                       | `SIGNAL_CLASSIF_LOWER_BOUND` / `_UPPER_BOUND`     | 10 / **75**     |
| Retrospective window *W*          | 2 to 30 days (minimal impact)   | `classify_signals(window_size=...)`               | 7 (demo)        |

Two implementation details matter when comparing with the paper:

- **Classification percentiles.** The bounds are module-level constants read
  from `bertrend/config/bertrend_default_config.toml` when `bertrend` is
  imported; the `signal_classif_*` keys of a per-instance BERTrend config are
  not used for classification. The script below sets them explicitly to the
  paper's P10/P50 (the current default upper bound is P75).
- **Decay unit.** The code applies `exp(-decay_factor * Δt ** decay_power)` with
  `decay_power = 2` (the paper's `e^{-λΔt²}`), where Δt is counted in **time
  slices** (`granularity` days each), whereas the paper text expresses Δt in
  days. With `G = 7`, using `decay_factor = 0.01 * 7**2` makes the code's
  decay match a per-day Δt.

## 3. Run BERTrend on the arXiv corpus

Install BERTrend (see the main README), preferably on a machine with a GPU, then
run the following script. Embedding ~370k abstracts with `all-mpnet-base-v2`
takes some time on a GPU, and much longer on CPU; caching the embeddings
(`np.save`) avoids recomputing them between runs.

```python
import json
from pathlib import Path

import pandas as pd

import bertrend.BERTrend as bertrend_module
from bertrend import DATA_PATH, MODELS_DIR
from bertrend.BERTopicModel import BERTopicModel
from bertrend.BERTrend import BERTrend
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.data_loading import TEXT_COLUMN, group_by_days, load_data, split_data

ARXIV_SNAPSHOT = Path("arxiv-metadata-oai-snapshot.json")
DATASET = DATA_PATH / "arxiv_cs_2017_2023.jsonl"
MODELS_PATH = MODELS_DIR / "arxiv_cs"
PRIMARY_ONLY = False  # True: keep only papers whose primary category is cs.*
GRANULARITY = 7  # days
WINDOW_SIZE = 7  # days, retrospective window W

# 1. Build a BERTrend-compatible dataset (`text` + `timestamp` columns)
rows = []
with ARXIV_SNAPSHOT.open() as f:
    for line in f:
        paper = json.loads(line)
        categories = paper["categories"].split()
        if PRIMARY_ONLY:
            is_cs = categories[0].startswith("cs.")
        else:
            is_cs = any(c.startswith("cs.") for c in categories)
        if not is_cs:
            continue
        rows.append(
            {
                "timestamp": paper["versions"][0]["created"],  # first submission (v1)
                "text": " ".join(paper["abstract"].split()),
                "title": " ".join(paper["title"].split()),
                "url": f"https://arxiv.org/abs/{paper['id']}",
            }
        )
df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df = df[(df["timestamp"] >= "2017-01-01") & (df["timestamp"] < "2024-01-01")]
df.to_json(DATASET, orient="records", lines=True, date_format="iso")
print(f"{len(df)} abstracts")  # paper: 367,248

# 2. Load and clean (adds the `document_id` and `source` columns);
#    abstracts are short, so they are not split by paragraph
df = load_data(DATASET, language="English")
df = split_data(df, split_by_paragraph="no")

# 3. Embed with the model used in the paper
embedding_service = EmbeddingService(
    local=True, model_name="sentence-transformers/all-mpnet-base-v2"
)
embeddings, _, _ = embedding_service.embed(texts=df[TEXT_COLUMN])

# 4. Topic model and BERTrend with the paper's hyperparameters
topic_model = BERTopicModel(
    {
        "global": {"language": "English"},
        "umap_model": {"n_neighbors": 15, "n_components": 5},
        "hdbscan_model": {"min_cluster_size": 2, "min_samples": 1},
        "vectorizer_model": {"ngram_range": [1, 2]},
    }
)
bertrend = BERTrend(
    config_file=f"""
granularity = {GRANULARITY}
min_similarity = 0.7
decay_factor = 0.01
decay_power = 2
signal_classif_lower_bound = 10
signal_classif_upper_bound = 50
""",
    topic_model=topic_model,
)
# Classification uses module-level percentile bounds: set the paper's P10/P50
bertrend_module.SIGNAL_CLASSIF_LOWER_BOUND = 10
bertrend_module.SIGNAL_CLASSIF_UPPER_BOUND = 50

# 5. Train one topic model per 7-day slice, merge them over time, compute popularity
grouped_data = group_by_days(df=df, day_granularity=GRANULARITY)
bertrend.train_topic_models(
    grouped_data=grouped_data,
    embedding_model=embedding_service.embedding_model_name,
    embeddings=embeddings,
    bertrend_models_path=MODELS_PATH,
)
bertrend.calculate_signal_popularity()
bertrend.save_model(models_path=MODELS_PATH)
```

The trained model is saved under `MODELS_PATH`; it can be reloaded later with
`BERTrend.restore_model(MODELS_PATH)` to build the figures without retraining.

## 4. Figure 2(b): signal types and topic counts over time

For each time slice, count the noise, weak and strong signals, the number of
topics found in that slice, and the number of topics after merging:

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

topic_models = bertrend.restore_topic_models(models_path=MODELS_PATH)
merged_counts = dict(bertrend.merge_df_size_over_time)

records = []
for ts in sorted(bertrend.doc_groups):
    noise_df, weak_df, strong_df = bertrend.classify_signals(WINDOW_SIZE, ts)
    n_topics = (
        topic_models[ts].topic_info_df["Topic"].max() + 1 if ts in topic_models else None
    )
    records.append(
        {
            "timestamp": ts,
            "noise": len(noise_df),
            "weak": len(weak_df),
            "strong": len(strong_df),
            "topics": n_topics,
            # no merge happens for the first slice: its own topics are the merged ones
            "topics_after_merging": merged_counts.get(ts, n_topics),
        }
    )
counts = pd.DataFrame(records).set_index("timestamp")
counts.to_csv("figure_2b_arxiv.csv")

fig = make_subplots(specs=[[{"secondary_y": True}]])
for column, name in [
    ("noise", "Number of Noise Signals"),
    ("weak", "Number of Weak Signals"),
    ("strong", "Number of Strong Signals"),
    ("topics", "Number of Topics"),
]:
    fig.add_trace(go.Scatter(x=counts.index, y=counts[column], name=name))
fig.add_trace(
    go.Scatter(
        x=counts.index,
        y=counts["topics_after_merging"],
        name="Number of Topics after merging",
    ),
    secondary_y=True,
)
fig.update_yaxes(title_text="Count", secondary_y=False)
fig.update_yaxes(title_text="Count After Merging", secondary_y=True)
fig.update_layout(title="arXiv cs.* papers | Jan 2017 to Dec 2023")
fig.show()
```

`BERTrend.save_signal_evolution_data(window_size, start_timestamp, end_timestamp)`
can also be used: it stores the per-slice noise/weak/strong dataframes as pickle
files under `SIGNAL_EVOLUTION_DATA_DIR`, from which the same counts can be
derived.

## 5. Figure 3(b): popularity of selected topics

Figure 3(b) follows three merged topics from June 2017 to October 2019. Topic
identifiers differ from one run to another, so look them up by their
representation, then plot their popularity on a log scale:

```python
from bertrend.trend_analysis.visualizations import create_topic_size_evolution_figure


def find_topics(keyword: str) -> list[int]:
    """Merged topics whose representation contains `keyword` at some point in time."""
    return [
        topic
        for topic, data in bertrend.topic_sizes.items()
        if any(keyword in representation for representation in data["Representations"])
    ]


for keyword in ["attention", "imagenet", "lstm"]:
    print(keyword, find_topics(keyword))

# Pick one topic per keyword from the lists above (e.g. the most popular one)
selected_topics = [...]

fig = create_topic_size_evolution_figure(bertrend.topic_sizes, topic_ids=selected_topics)
fig.update_yaxes(type="log", title_text="Popularity (Log Scale)")
fig.update_xaxes(range=["2017-06-01", "2019-10-31"])
fig.show()
```

To also display the noise/weak/strong threshold bands at a given date, as in the
demo, pass this figure to
`bertrend.trend_analysis.visualizations.plot_topic_size_evolution` together with
the window bounds and thresholds computed for that date.

## 6. Using the Streamlit demo instead

The *Weak signal analysis* demo can also load the dataset built in step 1:
since the file is written to `DATA_PATH` (`$BERTREND_BASE_DIR/data`), select it
in the **Data from server** tab, which has no upload size limit (see
[Working with large datasets](demos/weak_signals_demo.md#working-with-large-datasets)). Training on the full corpus from the
browser is however much less practical than the script above.
