# Article Scoring Service

BERTrend provides a dedicated FastAPI service to score article quality before topic modeling.
It exposes the logic implemented in `bertrend/article_scoring/scoring_agent.py` and returns:

- per-criterion quality metrics,
- an `overall_quality` level,
- and per-item processing errors when relevant.

This can be used to filter low-quality/noisy content upstream in data pipelines.

## Run the service

```bash
cd bertrend/services/article_scoring_server
python start.py
```

By default, the service reads its settings from:
`bertrend/services/article_scoring_server/config/default_config.toml`.

You can override the config file path with:

```bash
export ARTICLE_SCORING_API_CONFIG_FILE=/path/to/config.toml
python bertrend/services/article_scoring_server/start.py
```

The config defines:

- `host`
- `port`
- `number_workers`

## API endpoints

- `GET /` → redirect to Swagger docs (`/docs`)
- `GET /health` → basic liveness check (`{"status": "ok"}`)
- `POST /score` → score a batch of articles

Request body for `POST /score`:

```json
{
  "articles": [
    "First article text...",
    "Second article text..."
  ]
}
```

Response shape:

```json
{
  "results": [
    {
      "article_index": 0,
      "quality_metrics": {
        "scores": {
          "depth_of_reporting": 0.8,
          "originality_and_exclusivity": 0.6
        },
        "weights": {
          "depth_of_reporting": 1.0,
          "originality_and_exclusivity": 1.0
        },
        "assessment_summary": "...",
        "final_score": 0.63,
        "quality_level": "Average"
      },
      "overall_quality": "AVERAGE",
      "error": null
    }
  ]
}
```

If the scoring backend fails globally, `POST /score` returns HTTP `500`.
If only one item fails, the response still returns `200` and sets `error` for that item.
