# Scheduling Service

BERTrend provides a FastAPI-based scheduling service that manages periodic and one-off jobs using
[APScheduler](https://apscheduler.readthedocs.io/). Jobs are persisted in an SQLite database so they survive
service restarts.

A companion NiceGUI dashboard lets you monitor job history and execution timelines visually.

---

## Overview

| Component      | Technology                                     | Default port |
|----------------|------------------------------------------------|--------------|
| Scheduling API | FastAPI + APScheduler                          | **8882**     |
| Job dashboard  | NiceGUI                                        | **8885**     |
| Job store      | SQLite (`~/.bertrend/db/bertrend_jobs.sqlite`) | —            |
| Executor       | `ProcessPoolExecutor` (up to 100 workers)      | —            |
| Timezone       | `Europe/Paris`                                 | —            |

---

## Running the service

### Locally

From the project root:

```bash
python -m bertrend.services.scheduling.scheduling_service
```

Or with uvicorn directly:

```bash
uvicorn bertrend.services.scheduling.scheduling_service:app \
    --host 0.0.0.0 --port 8882 --workers 1
```

### Health check

```bash
curl http://localhost:8882/health
# {"status": "ok"}
```

The root path (`/`) redirects to the interactive API documentation at `/docs`.

### Dashboard

```bash
python bertrend/services/scheduling/dashboard_scheduling.py
```

Open `http://localhost:8885` in a browser to view job execution history and timelines.

---

## Trigger types

Three trigger types are supported when creating a job:

| `job_type` | Description                     | Required fields                                                                                                                         |
|------------|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `interval` | Repeat at a fixed interval      | At least one of `seconds`, `minutes`, `hours`, `days`                                                                                   |
| `cron`     | Run on a cron schedule          | `cron_expression` (5-part string) **or** individual `cron_minute` / `cron_hour` / `cron_day` / `cron_month` / `cron_day_of_week` fields |
| `date`     | Run once at a specific datetime | `run_date` (ISO 8601 datetime)                                                                                                          |

---

## API endpoints

### Info

| Method | Path      | Description         |
|--------|-----------|---------------------|
| `GET`  | `/health` | Liveness check      |
| `GET`  | `/`       | Redirect to `/docs` |

### Job functions

| Method | Path         | Description                                                                  |
|--------|--------------|------------------------------------------------------------------------------|
| `GET`  | `/functions` | List available built-in job functions with their signatures and descriptions |

### Job management

| Method   | Path                    | Description                                                       |
|----------|-------------------------|-------------------------------------------------------------------|
| `POST`   | `/jobs`                 | Create a new scheduled job                                        |
| `GET`    | `/jobs`                 | List all scheduled jobs                                           |
| `GET`    | `/jobs/{job_id}`        | Get details of a specific job                                     |
| `PUT`    | `/jobs/{job_id}`        | Update an existing job (trigger, function, args, kwargs, options) |
| `DELETE` | `/jobs/{job_id}`        | Remove a job                                                      |
| `POST`   | `/jobs/find`            | Search jobs by regex patterns on their `kwargs`                   |
| `POST`   | `/jobs/{job_id}/pause`  | Pause a job                                                       |
| `POST`   | `/jobs/{job_id}/resume` | Resume a paused job                                               |
| `POST`   | `/jobs/{job_id}/run`    | Execute a job immediately, outside its schedule                   |

### Cron utilities

| Method | Path             | Description                                                       |
|--------|------------------|-------------------------------------------------------------------|
| `POST` | `/cron/validate` | Validate a cron expression and preview the next 5 execution times |

---

## Data models

### `JobCreate` (request body for `POST /jobs`)

| Field                                                                        | Type             | Description                                                |
|------------------------------------------------------------------------------|------------------|------------------------------------------------------------|
| `job_id`                                                                     | `str`            | Unique identifier for the job                              |
| `job_name`                                                                   | `str` (optional) | Human-readable name (defaults to `job_id`)                 |
| `job_type`                                                                   | `str`            | `"interval"`, `"cron"`, or `"date"`                        |
| `function_name`                                                              | `str`            | Name of the built-in function to execute                   |
| `args`                                                                       | `list`           | Positional arguments passed to the function                |
| `kwargs`                                                                     | `dict`           | Keyword arguments passed to the function                   |
| `replace_existing`                                                           | `bool`           | Overwrite job if `job_id` already exists (default: `true`) |
| `seconds` / `minutes` / `hours` / `days`                                     | `int`            | Interval trigger fields                                    |
| `cron_expression`                                                            | `str`            | 5-part cron string, e.g. `"0 12 * * *"`                    |
| `cron_minute` / `cron_hour` / `cron_day` / `cron_month` / `cron_day_of_week` | `str`            | Individual cron fields                                     |
| `run_date`                                                                   | `datetime`       | One-shot execution datetime                                |
| `max_instances`                                                              | `int`            | Max concurrent instances of this job (default: `3`)        |
| `coalesce`                                                                   | `bool`           | Merge missed executions into one (default: `false`)        |

### `JobResponse`

| Field             | Type            | Description                               |
|-------------------|-----------------|-------------------------------------------|
| `job_id`          | `str`           | Job identifier                            |
| `name`            | `str`           | Job name                                  |
| `next_run_time`   | `datetime`      | Next scheduled execution (null if paused) |
| `trigger`         | `str`           | Human-readable trigger description        |
| `func`            | `str`           | Fully qualified function reference        |
| `args` / `kwargs` | `list` / `dict` | Arguments passed to the function          |
| `executor`        | `str`           | Executor name                             |
| `max_instances`   | `int`           | Max concurrent instances                  |

### `JobFindRequest` (request body for `POST /jobs/find`)

| Field             | Type   | Description                                                                                     |
|-------------------|--------|-------------------------------------------------------------------------------------------------|
| `kwargs_patterns` | `dict` | Keys are `kwargs` field names; values are regex strings or nested dicts for deep matching       |
| `match_all`       | `bool` | `true` = all patterns must match (AND); `false` = any match is sufficient (OR). Default: `true` |

---

## Built-in job functions

The service ships with two built-in functions registered in `JOB_FUNCTIONS`:

### `http_request`

Executes an HTTP request (similar to `curl`). Useful for triggering other BERTrend services on a schedule.

| Parameter   | Type   | Default | Description                |
|-------------|--------|---------|----------------------------|
| `url`       | `str`  | —       | Target URL                 |
| `method`    | `str`  | `"GET"` | HTTP method                |
| `headers`   | `dict` | `None`  | Request headers            |
| `json_data` | `dict` | `None`  | JSON body                  |
| `timeout`   | `int`  | `600`   | Request timeout in seconds |

### `sample_job`

A simple example function that logs a message. Useful for testing the scheduler.

| Parameter | Type  | Default             | Description    |
|-----------|-------|---------------------|----------------|
| `message` | `str` | `"Default message"` | Message to log |

> **Extending the function registry**: Add new module-level functions to
> `bertrend/services/scheduling/job_utils/job_functions.py` and register them in the `JOB_FUNCTIONS` dict.
> Functions **must** be defined at module level (not as lambdas or closures) because the `ProcessPoolExecutor`
> requires them to be picklable.

---

## Usage examples

### Create an interval job

```bash
curl -X POST http://localhost:8882/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "fetch-rss-every-hour",
    "job_type": "interval",
    "function_name": "http_request",
    "hours": 1,
    "kwargs": {
      "url": "http://localhost:8001/fetch",
      "method": "POST"
    }
  }'
```

### Create a cron job (daily at noon)

```bash
curl -X POST http://localhost:8882/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "daily-newsletter",
    "job_type": "cron",
    "function_name": "http_request",
    "cron_expression": "0 12 * * *",
    "kwargs": {
      "url": "http://localhost:8003/generate",
      "method": "POST"
    }
  }'
```

### Create a one-shot job

```bash
curl -X POST http://localhost:8882/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "one-time-report",
    "job_type": "date",
    "function_name": "http_request",
    "run_date": "2025-06-01T09:00:00",
    "kwargs": {"url": "http://localhost:8003/report", "method": "GET"}
  }'
```

### List all jobs

```bash
curl http://localhost:8882/jobs
```

### Pause / resume a job

```bash
curl -X POST http://localhost:8882/jobs/daily-newsletter/pause
curl -X POST http://localhost:8882/jobs/daily-newsletter/resume
```

### Run a job immediately

```bash
curl -X POST http://localhost:8882/jobs/daily-newsletter/run
```

### Find jobs by kwargs pattern

```bash
curl -X POST http://localhost:8882/jobs/find \
  -H "Content-Type: application/json" \
  -d '{
    "kwargs_patterns": {"url": ".*newsletter.*"},
    "match_all": true
  }'
```

### Validate a cron expression

```bash
curl -X POST http://localhost:8882/cron/validate \
  -H "Content-Type: application/json" \
  -d '{"expression": "0 12 * * 1-5"}'
```

Response includes a human-readable description and the next 5 execution times.

---

## Persistence and timezone

- Jobs are stored in an SQLite database at `~/.bertrend/db/bertrend_jobs.sqlite`.
- All schedules use the `Europe/Paris` timezone.
- Jobs survive service restarts: on startup the service reloads all persisted jobs and logs their next run times.

---

## Integration with BERTrend

When `SCHEDULER_SERVICE_TYPE=apscheduler` is set in the environment, BERTrend uses
`bertrend_apps/common/apscheduler_utils.py` to interact with this service over HTTP instead of the system
crontab. Set `SCHEDULER_SERVICE_URL` to point to the running instance:

```bash
SCHEDULER_SERVICE_TYPE=apscheduler
SCHEDULER_SERVICE_URL=http://localhost:8882/
```

See [`scheduler_configuration.md`](scheduler_configuration.md) for a full comparison of the crontab and
APScheduler approaches, Docker deployment instructions, and migration guidance.
