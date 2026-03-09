# Summary Service

BERTrend provides a dedicated FastAPI service to summarize one or multiple texts.

It exposes multiple summarization backends (LLM, extractive, enhanced) and returns:

- generated summaries,
- selected summarizer type,
- language,
- processing time in milliseconds.

## Run the service

```bash
python bertrend/services/summary_server/start.py
```

By default, the service reads its settings from:
`bertrend/services/summary_server/config/default_config.toml`.

You can override the config file path with:

```bash
export SUMMARY_API_CONFIG_FILE=/path/to/config.toml
python bertrend/services/summary_server/start.py
```

The config defines:

- `host`
- `port`
- `number_workers`

## API endpoints

- `GET /` → redirect to Swagger docs (`/docs`)
- `GET /health` → basic liveness check (`{"status": "ok"}`)
- `GET /summarizers` → list available summarizer backends and requirements
- `POST /summarize` → summarize a text or a batch of texts

Request body for `POST /summarize`:

```json
{
  "text": [
    "First article text...",
    "Second article text..."
  ],
  "summarizer_type": "extractive",
  "language": "fr",
  "max_sentences": 5,
  "max_length_ratio": 0.5,
  "max_words": 120
}
```

Response shape:

```json
{
  "summaries": [
    "Summary 1",
    "Summary 2"
  ],
  "summarizer_type": "extractive",
  "language": "fr",
  "processing_time_ms": 123.45
}
```

If `summarizer_type` is unknown, `POST /summarize` returns HTTP `400` with the list of available values.
