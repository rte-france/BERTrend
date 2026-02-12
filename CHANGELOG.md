## v0.4.1 - 2026-02-12

### Features

- Added a universal summarization service with FastAPI-based server and client implementation.

## v0.4.0 - 2026-02-09

### Features

- Queue monitoring dashboard and UI enhancements, including improved messages and timestamps.
- RabbitMQ queue service updated to use `aio-pika` for asynchronous processing, with revised queue management.
- Added priority handling and per-queue TTL differentiation.
- Modernized OpenAI client parsing with OpenAI Agents SDK compatibility (including GPT-5 support).

### Fixes

- Scheduler initialization now avoids multiple `BackgroundScheduler` instances.
- Timezone handling for timestamp display.
- Message formatting and JSON serialization adjustments.
- Docker user configuration for scheduling service.
- Tests and missing configuration fixes.

### Refactors

- Moved `bertrend_apps` inside `bertrend` and reorganized imports/classes.
- Code cleanup and ruff formatting passes.

### Chore

- Updated copyright.
- Project guidelines update and version bump metadata.

## v0.3.24 - 2026-01-28

### Fixes

- Fixed scheduler jobs and avoided multiple BackgroundScheduler instances.
- Added missing dependencies for scheduler.

### Refactors

- Replaced Black with Ruff for Python code formatting.

## v0.3.23 - 2025-11-24

### Features

- Added "who is connected" indicator in UI.
- Added next execution time for scrapping and learning in the UI.

### Fixes

- Improved management of request connections (one session per HTTP connection, thread-safe).
- Better handling of timeouts and URL decoding issues.
- Fixed translation issues in `clickable_df`.
- Fixed Pydantic compatibility and newscatcher API library change.
- Fixed various edge cases generating execution failures.

### Refactors

- Optimized requests.Session initialization to be thread-safe in APSchedulerUtils.

## v0.3.22 - 2025-11-04

### Features

- Added `HOST_UID` and `HOST_GID` to Docker configuration.

## v0.3.21 - 2025-11-03

### Features

- Integrated BERTrend apps service and Scheduler service.
- Improved scripts and added parsing of Google News URLs in RSS feeds.

### Fixes

- Resolved Docker networking issues and environment variable handling.
- Fixed loguru coloring with additional sinks.

### Refactors

- Simplified data provider API and reorganized test structure.

## v0.3.20 - 2025-10-20

### Features

- Introduced new visualizations and a feed data overview.
- Added support for `.env` files for environment variable management.
- Parallelized data loading for performance speedup.
- Introduced a new Python scheduler as an alternative to crontab.

### Refactors

- Updated environment variables and harmonized parameter names.
- Simplified return types and harmonized CUDA device variables.
- Asynchronous requests to the data provider service.

## v0.3.19 - 2025-10-14

### Features

- Added RSS data provider.
- Automatic report sending after generation (configurable in GUI).

### Fixes

- Improved language handling for topic titles and summaries.
- Fixed interpretation data loading and file paths.

### Chore

- Cache improvements and Streamlit component updates.

## v0.3.18 - 2025-10-01

### Features

- Simplified Dockerfiles and improved Docker configuration for BERTrend.
- Added GitHub Actions for publication of Docker images.

### Fixes

- Added log rotation and healthchecks.
- Fixed communication problems between containers through proxies.
- Optimized Docker image size and fixed volume paths.

## v0.3.17 - 2025-09-25

### Features

- Significant prompt improvements for LLM interactions.

### Fixes

- Fixed output formatting issues.

## v0.3.16 - 2025-09-23

### Fixes

- Fixed model configuration to use the same language as the feed.

## v0.3.15 - 2025-09-23

### Features

- Performance improvements in core processing.
- Added test user for demo application.