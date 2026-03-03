# Article Scoring API Reference

## Module: `bertrend/article_scoring/`

An LLM-powered article quality scoring system that evaluates articles against configurable criteria and produces
structured quality assessments.

---

## Data Models (`article_scoring.py`)

### `QualityLevel` (Enum)

Ordinal quality levels for scored articles:

| Value           | Label         |
|-----------------|---------------|
| `POOR`          | Poor quality  |
| `BELOW_AVERAGE` | Below average |
| `AVERAGE`       | Average       |
| `GOOD`          | Good          |
| `EXCELLENT`     | Excellent     |

Supports comparison (`<`, `==`) and conversion from string via `QualityLevel.from_string(label)`.

### `CriteriaScores`

Pydantic model holding per-criterion scores (each 0–10).

```python
class CriteriaScores(BaseModel):
    relevance: float
    depth: float
    credibility: float
    clarity: float
    timeliness: float
    originality: float
    actionability: float
    ...
```

#### Methods

| Method                  | Returns     | Description                      |
|-------------------------|-------------|----------------------------------|
| `get_criterion_names()` | `list[str]` | List all scored criterion names. |
| `to_dict()`             | `dict`      | Export scores as a dictionary.   |

### `WeightConfig`

Pydantic model for criterion weights. All weights must sum to 1.0.

```python
class WeightConfig(BaseModel):
    relevance: float = 0.20
    depth: float = 0.15
    credibility: float = 0.15
    ...
```

### `ArticleScore`

The main scoring result container.

```python
class ArticleScore(BaseModel):
    title: str
    criteria_scores: CriteriaScores
    weights: WeightConfig
    justification: str
```

#### Key Properties & Methods

| Member                     | Returns        | Description                                          |
|----------------------------|----------------|------------------------------------------------------|
| `final_score`              | `float`        | Weighted average score (0–10).                       |
| `quality_level`            | `QualityLevel` | Mapped quality level from final score.               |
| `get_detailed_breakdown()` | `list[dict]`   | Per-criterion breakdown with weighted contributions. |
| `get_top_strengths(n)`     | `list`         | Top N highest-scoring criteria.                      |
| `get_top_weaknesses(n)`    | `list`         | Top N lowest-scoring criteria.                       |
| `to_report()`              | `str`          | Human-readable text report.                          |
| `export_to_dict()`         | `dict`         | Full export including computed fields.               |
| `to_json()`                | `str`          | JSON serialization.                                  |

---

## Scoring Agent (`scoring_agent.py`)

### `score_articles(articles, model, max_concurrency)`

Score a batch of articles using an LLM agent.

| Parameter         | Type         | Default | Description                                             |
|-------------------|--------------|---------|---------------------------------------------------------|
| `articles`        | `list[dict]` | —       | Articles to score (each with `title`, `content`, etc.). |
| `model`           | `str`        | `None`  | LLM model name. Uses `OPENAI_DEFAULT_MODEL` if `None`.  |
| `max_concurrency` | `int`        | `5`     | Maximum parallel LLM requests.                          |

**Returns:** `list[ArticleScore]` — Scored articles.

Uses `AsyncAgentConcurrentProcessor` from `llm_utils.agent_utils` internally.

---

## Prompts (`prompts.py`)

Contains the system and user prompt templates that instruct the LLM on how to evaluate articles. The prompts define:

- Evaluation criteria and their descriptions
- Scoring scale (0–10)
- Expected output format (structured JSON matching `CriteriaScores`)

---

## Configuration

The scoring system reads LLM configuration from environment variables:

- `OPENAI_API_KEY` — API key for the LLM provider.
- `OPENAI_BASE_URL` — Base URL (supports OpenAI-compatible proxies).
- `OPENAI_DEFAULT_MODEL` — Default model for scoring.

Custom weights can be passed by constructing a `WeightConfig` and providing it when creating `ArticleScore` instances.
