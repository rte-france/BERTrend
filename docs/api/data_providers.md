# Data Providers API Reference

## Module: `bertrend/bertrend_apps/data_provider/`

Data providers are adapters that fetch articles from various external sources. All providers implement the
`DataProvider` base class interface.

---

## Base Class (`data_provider.py`)

### `DataProvider` (Abstract)

All data providers inherit from this base class and implement the `get_articles()` method.

```python
class DataProvider(ABC):
    def get_articles(self, query: str, **kwargs) -> pd.DataFrame
```

**Returns:** A DataFrame with standardized columns: `title`, `content`, `url`, `source`, `date`, etc.

---

## Available Providers

### RSS Feed Provider (`rss_feed_provider.py`)

Fetches articles from RSS feeds using `feedparser`.

- **Input:** RSS feed URL(s).
- **Features:** Supports multiple feeds, date filtering, content extraction.

### Atom Feed Provider (`atom_feed_provider.py`)

Fetches articles from Atom feeds.

- **Input:** Atom feed URL(s).
- **Features:** Similar to RSS but handles Atom-specific XML structure.

### ArXiv Provider (`arxiv_provider.py`)

Fetches academic papers from the ArXiv API.

- **Input:** Search query, category filters.
- **Features:** Retrieves abstracts, authors, categories, and PDF links.

### Google News Provider (`google_news_provider.py`)

Fetches news articles via Google News.

- **Input:** Search query, language, region.
- **Features:** Keyword-based news search with language/region filtering.

### Bing News Provider (`bing_news_provider.py`)

Fetches news articles via the Bing News Search API.

- **Input:** Search query, API key.
- **Features:** Requires a Bing Search API key. Supports market and freshness filters.

### Newscatcher Provider (`newscatcher_provider.py`)

Fetches news articles via the Newscatcher API.

- **Input:** Search query, API key.
- **Features:** Broad news coverage with topic and language filtering.

### Deep Research Provider (`deep_research_provider.py`)

An AI-powered research provider that uses an LLM agent to perform multi-step deep research on a topic.

#### Architecture

1. **Planning phase** — An LLM generates a `ResearchPlan` with sub-queries.
2. **Research phase** — Each sub-query is executed concurrently using DuckDuckGo news search via a `function_tool`.
3. **Aggregation phase** — Results are deduplicated and compiled into a standardized DataFrame.

#### Classes

##### `ResearchPlan`

Pydantic model for the LLM-generated research plan.

| Field         | Type        | Description                                 |
|---------------|-------------|---------------------------------------------|
| `sub_queries` | `list[str]` | Generated sub-queries to research.          |
| `reasoning`   | `str`       | LLM's reasoning for the chosen sub-queries. |

##### `SubQueryResult`

Pydantic model for results from a single sub-query.

| Field      | Type         | Description                        |
|------------|--------------|------------------------------------|
| `articles` | `list[dict]` | Articles found for this sub-query. |
| `summary`  | `str`        | Brief summary of findings.         |

##### `DeepResearchProvider`

```python
DeepResearchProvider(
    model: str = DEFAULT_MODEL,
    max_sub_queries: int = 5,
    search_delay: float = 3.0,
    language_code: str = "us-en",
)
```

| Parameter         | Type    | Default                | Description                                        |
|-------------------|---------|------------------------|----------------------------------------------------|
| `model`           | `str`   | `OPENAI_DEFAULT_MODEL` | LLM model for planning and research.               |
| `max_sub_queries` | `int`   | `5`                    | Maximum number of sub-queries to generate.         |
| `search_delay`    | `float` | `3.0`                  | Delay between DuckDuckGo requests (rate limiting). |
| `language_code`   | `str`   | `"us-en"`              | Language/region code for news search.              |

#### Key Methods

| Method                             | Description                                                        |
|------------------------------------|--------------------------------------------------------------------|
| `get_articles(query, **kwargs)`    | Run the full research pipeline and return a DataFrame of articles. |
| `_plan(query)`                     | Generate a research plan with sub-queries.                         |
| `_research_sub_query(sub_query)`   | Execute a single sub-query research task.                          |
| `_research_all_async(sub_queries)` | Execute all sub-queries concurrently.                              |
| `_run_research_pipeline(query)`    | Orchestrate the full plan → research → aggregate pipeline.         |
| `_aggregate_articles(results)`     | Deduplicate and merge articles from all sub-queries.               |

#### Usage Example

```python
from bertrend.bertrend_apps.data_provider.deep_research_provider import DeepResearchProvider

provider = DeepResearchProvider(model="gpt-4.1-mini", max_sub_queries=3)
articles_df = provider.get_articles("renewable energy trends 2026")
```

---

## Utility Modules

### `data_provider_utils.py`

High-level orchestration functions used by the service layer:

| Function                               | Description                                                |
|----------------------------------------|------------------------------------------------------------|
| `scrape(request)`                      | Execute a scraping request using the appropriate provider. |
| `auto_scrape(request)`                 | Run automated scraping with scheduling logic.              |
| `scrape_feed_from_config(config_path)` | Scrape from a feed configuration TOML file.                |
| `generate_query_file(request)`         | Generate a query configuration file for a topic.           |

### `utils.py`

Shared utilities:

| Function        | Description                                      |
|-----------------|--------------------------------------------------|
| `wait(seconds)` | Async-compatible delay (used for rate limiting). |
