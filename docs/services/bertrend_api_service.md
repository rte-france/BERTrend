# BERTrend API Service

BERTrend provides a FastAPI service for high-level operations such as scraping, training, report generation, and
newsletter workflows.

This service is defined in `bertrend/bertrend_apps/services/bertrend/` and dispatches requests to RabbitMQ-backed
workers.

## Run the service

```bash
python bertrend/bertrend_apps/services/bertrend/start.py
```

By default, the service reads its settings from:
`bertrend/bertrend_apps/services/bertrend/config/default_config.toml`.

You can override the config file path with:

```bash
export BERTREND_API_CONFIG_FILE=/path/to/config.toml
python bertrend/bertrend_apps/services/bertrend/start.py
```

The config defines:

- `host`
- `port`
- `number_workers`

## API endpoints

The service exposes several routers:

- info endpoints (`/health`, `/num_workers`, `/`)
- data provider endpoints (`/scrape`, `/auto_scrape`, `/generate_query_file`, `/scrape_from_feed`,
  `/automate_scrapping`)
- newsletters endpoints (`/newsletter_from_feed`, `/automate_newsletter`)
- BERTrend app endpoints (`/train_new`, `/regenerate`, `/generate_report`)

For the complete endpoint-level request/response reference, see `docs/api/services.md`.
