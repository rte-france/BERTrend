# Services & Infrastructure API Reference

## FastAPI Service Layer

**Location:** `bertrend/bertrend_apps/services/bertrend/`

The BERTrend API is a FastAPI application that exposes endpoints for data scraping, model training, report generation,
and newsletter automation. Requests are dispatched to a RabbitMQ queue and processed asynchronously by workers.

---

### Configuration (`config/settings.py`)

#### `BERTrendAPIConfig`

Pydantic settings model for the API server.

| Field            | Type  | Description                      |
|------------------|-------|----------------------------------|
| `host`           | `str` | Bind address for the API server. |
| `port`           | `int` | Port number.                     |
| `number_workers` | `int` | Number of Uvicorn workers.       |

#### `get_config()`

Load configuration from a TOML file. The file path can be overridden via the `BERTREND_API_CONFIG_FILE` environment
variable; otherwise, `config/default_config.toml` is used.

---

### Routers

#### BERTrend App Router (`routers/bertrend_app.py`)

| Endpoint           | Method | Description                                                               |
|--------------------|--------|---------------------------------------------------------------------------|
| `/train_new`       | POST   | Train a new topic model on incoming data. Accepts `TrainNewModelRequest`. |
| `/regenerate`      | POST   | Regenerate models from existing data. Accepts `RegenerateRequest`.        |
| `/generate_report` | POST   | Generate an automated analysis report. Accepts `GenerateReportRequest`.   |

#### Data Provider Router (`routers/data_provider.py`)

| Endpoint               | Method | Description                                                                      |
|------------------------|--------|----------------------------------------------------------------------------------|
| `/scrape`              | POST   | Scrape articles from configured sources. Accepts `ScrapeRequest`.                |
| `/auto_scrape`         | POST   | Run automated scraping with scheduling. Accepts `AutoScrapeRequest`.             |
| `/generate_query_file` | POST   | Generate a query configuration file. Accepts `GenerateQueryFileRequest`.         |
| `/scrape_from_feed`    | POST   | Scrape articles from a specific feed configuration. Accepts `ScrapeFeedRequest`. |
| `/automate_scrapping`  | POST   | Set up automated recurring scraping.                                             |

#### Newsletters Router (`routers/newsletters.py`)

| Endpoint                | Method | Description                                                     |
|-------------------------|--------|-----------------------------------------------------------------|
| `/newsletter_from_feed` | POST   | Generate a newsletter from a feed. Accepts `NewsletterRequest`. |
| `/automate_newsletter`  | POST   | Set up automated newsletter generation.                         |

#### Info Router (`routers/info.py`)

| Endpoint       | Method | Description                              |
|----------------|--------|------------------------------------------|
| `/health`      | GET    | Health check endpoint.                   |
| `/num_workers` | GET    | Return the number of configured workers. |
| `/`            | GET    | Root endpoint with API information.      |

---

### Request/Response Models (`models/`)

#### `bertrend_app_models.py`

| Model                   | Fields                                            | Description                   |
|-------------------------|---------------------------------------------------|-------------------------------|
| `StatusResponse`        | `status`, `message`                               | Generic status response.      |
| `TrainNewModelRequest`  | `reference_timestamp`, `data`, `models_path`, ... | Request to train a new model. |
| `TrainNewModelResponse` | `status`, `model_path`                            | Training result.              |
| `RegenerateRequest`     | `models_path`, `granularity`, `language`, ...     | Request to regenerate models. |
| `GenerateReportRequest` | `models_path`, `window_size`, `language`, ...     | Request to generate a report. |

#### `data_provider_models.py`

| Model                       | Fields                                             | Description                   |
|-----------------------------|----------------------------------------------------|-------------------------------|
| `ScrapeRequest`             | `query`, `sources`, `language`, `max_results`, ... | Scraping parameters.          |
| `ScrapeResponse`            | `status`, `num_articles`                           | Scraping result.              |
| `AutoScrapeRequest`         | `query`, `sources`, `schedule`, ...                | Automated scraping config.    |
| `GenerateQueryFileRequest`  | `topic`, `language`                                | Query file generation params. |
| `GenerateQueryFileResponse` | `status`, `file_path`                              | Query file result.            |
| `ScrapeFeedRequest`         | `feed_config_path`                                 | Feed-based scraping params.   |

#### `newsletters_models.py`

| Model               | Fields                         | Description                       |
|---------------------|--------------------------------|-----------------------------------|
| `NewsletterRequest` | `feed_config`, `language`, ... | Newsletter generation parameters. |

---

## Queue Management (`services/queue_management/`)

### BERTrend Worker (`bertrend_worker.py`)

The worker consumes messages from a RabbitMQ queue and dispatches them to the appropriate handler function.

#### Class: `BertrendWorker`

```python
BertrendWorker(
    rabbitmq_url: str,
    queue_name: str,
)
```

| Method                     | Description                                                                            |
|----------------------------|----------------------------------------------------------------------------------------|
| `start()`                  | Connect to RabbitMQ and begin consuming messages.                                      |
| `process_request(message)` | Parse an incoming message and route it to the correct handler.                         |
| `callback(message)`        | aio_pika callback that wraps `process_request` with error handling and acknowledgment. |

#### Handler Functions

Each handler deserializes the request body and calls the corresponding core logic:

| Handler                             | Processes                  | Core Function                 |
|-------------------------------------|----------------------------|-------------------------------|
| `handle_scrape(body)`               | `ScrapeRequest`            | `scrape()`                    |
| `handle_auto_scrape(body)`          | `AutoScrapeRequest`        | `auto_scrape()`               |
| `handle_scrape_feed(body)`          | `ScrapeFeedRequest`        | `scrape_feed_from_config()`   |
| `handle_generate_query_file(body)`  | `GenerateQueryFileRequest` | `generate_query_file()`       |
| `handle_train_new(body)`            | `TrainNewModelRequest`     | `train_new_model()`           |
| `handle_regenerate(body)`           | `RegenerateRequest`        | `regenerate_models()`         |
| `handle_generate_report(body)`      | `GenerateReportRequest`    | `generate_automated_report()` |
| `handle_generate_newsletters(body)` | `NewsletterRequest`        | `process_newsletter()`        |

#### `main()`

Entry point that creates a `BertrendWorker` and starts consuming. Reads RabbitMQ connection details from
environment/config.

---

## Queue Monitor (`services/queue_monitor/app.py`)

A Streamlit-based monitoring dashboard for inspecting RabbitMQ queues used by BERTrend.

### Class: `RabbitMQAdminClient`

HTTP client for the RabbitMQ Management API.

```python
RabbitMQAdminClient(
    host: str,
    port: int,
    username: str,
    password: str,
    vhost: str = "/",
)
```

| Method                             | Description                                             |
|------------------------------------|---------------------------------------------------------|
| `get_queue(queue_name)`            | Fetch queue metadata (message count, consumers, state). |
| `peek_messages(queue_name, count)` | Retrieve messages from a queue without consuming them.  |

### Streamlit UI Functions

| Function                           | Description                                                      |
|------------------------------------|------------------------------------------------------------------|
| `queue_icon(state)`                | Return an emoji icon based on queue state (idle, active, error). |
| `_decode_message_payload(payload)` | Decode and pretty-print a RabbitMQ message payload.              |
| `_format_timestamp(ts)`            | Format a Unix timestamp for display.                             |
| `render_queue_config_grid(queues)` | Render a grid of queue status cards in the Streamlit UI.         |

### Running the Monitor

```bash
streamlit run bertrend/bertrend_apps/services/queue_monitor/app.py
```

Requires RabbitMQ Management plugin to be enabled and accessible.

---

## Logging Utilities (`services/utils/logging_utils.py`)

### `get_file_logger(id, user_name, model_id)`

Create a dedicated log file for a specific operation and return a `loguru` handler ID.

| Parameter   | Type  | Default | Description                                         |
|-------------|-------|---------|-----------------------------------------------------|
| `id`        | `str` | —       | Operation identifier (e.g., `"train"`, `"scrape"`). |
| `user_name` | `str` | `""`    | Optional user name for log file organization.       |
| `model_id`  | `str` | `""`    | Optional model ID for log file organization.        |

**Returns:** `int` — The `loguru` handler ID (can be used to remove the handler later with `logger.remove(handler_id)`).

**Log file path:** `{BERTREND_LOG_PATH}/{user_name}/{model_id}/{id}_{user_name}_{model_id}_{timestamp}.log`

The log format includes timestamp, level, module, function, line number, and message with color support.
