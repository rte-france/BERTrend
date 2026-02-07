# Scheduler Configuration (APScheduler service vs system crontab)

BERTrend can schedule periodic jobs in two ways:

- **System crontab (default)**: Simple and robust on a traditional Linux host.
- **Internal APScheduler service (FastAPI)**: Recommended on Windows or when running in containers or when you want a self-contained scheduler you can manage with Docker Compose or Kubernetes. Jobs are persisted in an SQLite DB.

If you use the APScheduler service in a transient container, the scheduler will stop with the container. To mimic crontab behavior you must run the scheduler in an "always on" container with restart policy and persistent storage.

This document explains how to run the APScheduler service both locally and in Docker, and how to switch BERTrend to use it.

---

## 1. Switching between crontab and APScheduler

Configuration is via environment variables (see `.env_template`):

- `SCHEDULER_SERVICE_TYPE=crontab` or `apscheduler`
- `SCHEDULER_SERVICE_URL=http://scheduler:8000/` (when using the service)

When `SCHEDULER_SERVICE_TYPE=apscheduler`, BERTrend will use `apscheduler_utils.py` to interact with the service over HTTP.

---

## 2. Running the scheduler service locally

The scheduler service is a FastAPI app defined at `bertrend.services.scheduling.scheduling_service:app` and stores jobs in `data/jobs.sqlite`.

### Prerequisites

Python environment with project deps installed (`uvicorn`, `fastapi`, `apscheduler`, `sqlalchemy`). If missing:

```bash
pip install "uvicorn[standard]" fastapi apscheduler sqlalchemy
```

### Run the service

From the project root:

```bash
uvicorn bertrend.services.scheduling.scheduling_service:app --host 0.0.0.0 --port 8000 --proxy-headers
```

### Verify it responds

```bash
curl http://localhost:8000/
```

You should see a JSON payload describing the API and environment.

### Set BERTrend to use the local service

```bash
export SCHEDULER_SERVICE_TYPE=apscheduler
export SCHEDULER_SERVICE_URL=http://localhost:8000/
```

---

## 3. Running in Docker (lightweight image, always-on container)

To mimic crontab behavior, run the scheduler as a dedicated, always-on container with persistent storage and a restart policy.

**Preferred approach**: Build the lightweight scheduler image defined at `bertrend/services/scheduling/Dockerfile` (this image contains only the FastAPI scheduler service and its minimal runtime).

### Build the lightweight image

From the repo root:

```bash
docker build -f bertrend/services/scheduling/Dockerfile -t bertrend-scheduler:light .
```

### Start the scheduler container

Using that image:

```bash
docker run -d \
  --name bertrend-scheduler \
  -p 8000:8000 \
  --restart unless-stopped \
  -v $(pwd)/bertrend/services/scheduling/data:/app/data \
  -e TZ=Europe/Paris \
  bertrend-scheduler:light \
  uvicorn bertrend.services.scheduling.scheduling_service:app --host 0.0.0.0 --port 8000 --proxy-headers
```

### Notes

- Mount `/app/data` to persist the SQLite database (`jobs.sqlite`) across restarts. The example maps the repo's `bertrend/services/scheduling/data` directory; you can map any host directory, e.g., `$(pwd)/data`.
- The restart policy keeps the service always on, even after host reboots.
- Exposing port 8000 allows other containers/hosts to reach the scheduler.
- For a more complete guide (including docker-compose), see `bertrend/services/scheduling/Docker_setup.md`.

### Set clients to use it

```bash
SCHEDULER_SERVICE_TYPE=apscheduler
SCHEDULER_SERVICE_URL=http://localhost:8000/  # or http://scheduler:8000/ if using docker-compose networking
```

---

## 4. Verifying the service and basic API

### Root endpoint

```bash
curl http://<host>:8000/
```

### List available built-in functions

```bash
curl http://<host>:8000/functions
```

---

## 5. Using the service from BERTrend

When `SCHEDULER_SERVICE_TYPE=apscheduler` is set, BERTrend uses `bertrend_apps/common/apscheduler_utils.py`.

Typical flows that schedule tasks (scraping or newsletters) will call the service:

- The job content is persisted in SQLite (`data/jobs.sqlite`) by the service.
- Logs of the job commands referenced by the utils continue to be written under `BERTREND_LOG_PATH` in the main application.

Make sure the environment variable `SCHEDULER_SERVICE_URL` is reachable from the process that triggers scheduling. For docker-compose, `http://scheduler:8000/` is the common service URL.

---

## 6. Timezone and persistence

- The service runs with timezone `Europe/Paris` by default (configured in `scheduling_service.py`). If your deployment needs another timezone, either change it in code or run the container with `TZ` and adjust the service code accordingly.
- Job storage uses SQLite at `/app/data/jobs.sqlite`. Always mount `/app/data` to persist jobs.

---

## 7. Migrating from crontab

1. Set `SCHEDULER_SERVICE_TYPE=apscheduler` and `SCHEDULER_SERVICE_URL` to the running service.
2. Existing code continues to use the same helper API, but calls the service over HTTP under the hood.
3. Regex-based discovery/removal still works: the service includes the original command text in job name/metadata for compatibility.
