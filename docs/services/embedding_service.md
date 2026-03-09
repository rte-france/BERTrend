# Embedding Service

BERTrend provides a FastAPI embedding service used to generate vector embeddings for input texts.

This service wraps a `SentenceTransformer` model and exposes authenticated endpoints.

## Run the service

```bash
cd bertrend/services/embedding_server
python start.py
```

By default, the service reads its settings from:
`bertrend/services/embedding_server/config/default_config.toml`.

You can override the config file path with:

```bash
export EMBEDDING_API_CONFIG_FILE=/path/to/config.toml
python bertrend/services/embedding_server/start.py
```

The config defines:

- `host`
- `port`
- `number_workers`
- `model_name`
- `cuda_visible_devices`

## Authentication and authorization

The service uses OAuth2 token authentication.

- Use `POST /token` to obtain an access token.
- Use the returned bearer token for protected endpoints.
- `POST /encode` requires `full_access` scope.
- Admin endpoints (`/list_registered_clients`, `/rate-limits`) require `admin` scope.

## API endpoints

- `GET /health` → basic liveness check
- `GET /model_name` → currently loaded embedding model
- `GET /num_workers` → configured number of workers
- `POST /token` → generate access token
- `GET /list_registered_clients` → list registered OAuth clients (admin scope)
- `GET /rate-limits` → inspect rate-limit usage (admin scope)
- `POST /encode` → compute embeddings for one or multiple texts (`full_access` scope)

Request body for `POST /encode`:

```json
{
  "text": [
    "First text",
    "Second text"
  ],
  "show_progress_bar": false
}
```

Response shape:

```json
{
  "embeddings": [
    [0.01, -0.22, 0.58],
    [0.11, -0.15, 0.49]
  ]
}
```
