# Running BERTrend with Docker

This guide explains how to use BERTrend with Docker, which provides an easy way to run the application without installing dependencies directly on your system.

## Docker Images Overview

BERTrend provides these Docker images:

1. **Main BERTrend Image** (`bertrend:latest`): Contains the core BERTrend application and three demo applications:
   - Topic Analysis Demo (port 8083)
   - Weak Signals Demo (port 8084)
   - Prospective Demo (port 8081)

   It also runs, inside the same container (via supervisor), the FastAPI apps
   service (8881), the summary service (8886), the article scoring service
   (8887), the queue monitoring dashboard (8091) and the queue **workers** that
   the prospective demo relies on.

2. **Embedding Server Image** (`bertrend-embedding-server:latest`): Provides embedding services for the main application, running on port 6464.

3. **Scheduler Image** (`bertrend-scheduler:latest`, built from `bertrend/services/scheduling`): APScheduler service that fires scheduled jobs — feed scraping, model training, report generation. It listens on port 8000 in the container, published on the host as **8882**.

   The same image also backs the **Job Viewer** (`job_viewer` service, port
   8885): a NiceGUI dashboard over the scheduler API. It is a second entrypoint
   into that source tree — `dashboard_scheduling.py` needs only `requests` and
   `nicegui` — so it needs no image of its own. It reads the job list from
   `SCHEDULER_JOBS_URL`, set to `http://scheduler:8000/jobs` by Compose so it
   goes over the Compose network rather than back out through the public host.

The main and embedding-server images are built with NVIDIA CUDA support for GPU acceleration.

### Services and startup order

The prospective demo's automation needs two supporting services, which the Compose files now start for you:

- **RabbitMQ** (`rabbitmq:4.2-management`): the execution-queue broker the in-container workers consume from.
- **Scheduler** (`bertrend-scheduler`): registers/fires scheduled jobs, which call back to the app's FastAPI service (port 8881) to enqueue work onto RabbitMQ.

The Job Viewer (`job_viewer`) starts after the scheduler and only reads from it.

They must come up in order — **RabbitMQ first, then the scheduler, then the app**. The Compose files enforce this with `depends_on` health conditions (`rabbitmq` healthy → `scheduler` healthy → `bertrend`), so a plain `docker compose up -d` brings everything up in the right order. The app is wired to them via `RABBITMQ_HOST=rabbitmq`, `SCHEDULER_SERVICE_TYPE=apscheduler`, `SCHEDULER_SERVICE_URL=http://scheduler:8000/`, and `BERTREND_APPS_SERVICE_URL=http://bertrend:8881/`.

## Published ports

Every service the stack runs is published on **all interfaces**, so they are
reachable as `http://<host>:<port>/` — the way they were exposed before the
stack was containerised. There is no authentication in front of them, so on a
shared machine restrict access at the firewall, or re-add a `127.0.0.1:` prefix
to the port mappings and reach them through an SSH tunnel.

| Port | Service | Container |
|------|---------|-----------|
| 8081 | Prospective demo | `bertrend` |
| 8083 | Topic Analysis demo | `bertrend` |
| 8084 | Weak Signals demo | `bertrend` |
| 8091 | Queue monitoring dashboard | `bertrend` |
| 8881 | `bertrend_apps` FastAPI service (data scraping) | `bertrend` |
| 8886 | Summary service | `bertrend` |
| 8887 | Article scoring service | `bertrend` |
| 8882 | Scheduler API / docs (container port 8000) | `scheduler` |
| 8885 | Job Viewer dashboard | `job_viewer` |
| 15672 | RabbitMQ management UI | `rabbitmq` |
| 6464 | Embedding server (full stack only) | `embedding_server` |

RabbitMQ's AMQP port (5672) is deliberately **not** published: only the
in-container workers use it, over the Compose network.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) for running multi-container applications
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (optional, for GPU support)

## Quick Start with Docker Compose

The easiest way to run BERTrend is using Docker Compose, which will start both the main application and the embedding server:

1. Clone the BERTrend repository:
   ```bash
   git clone https://github.com/rte-france/BERTrend.git
   cd BERTrend
   ```

2. Create a `.env` file with your configuration (optional):
   - You can reuse the `.env` template at the repository root and fill in your values.
   - Note: When running outside Docker, BERTrend auto-loads the repo `.env` if `python-dotenv` is installed.
   - `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `OPENAI_DEFAULT_MODEL` are read from this
     file only (`env_file`), so values exported in your shell for other tools do not
     reach the containers. `OPENAI_DEFAULT_MODEL` has no default and must be set.
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_openai_endpoint_or_base_url
OPENAI_DEFAULT_MODEL=gpt-5.6-luna
BERTREND_BASE_DIR=/path/to/your/data/directory
```

3. Start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access the demos at:
   - Topic Analysis: http://localhost:8083
   - Weak Signals: http://localhost:8084
   - Prospective Demo: http://localhost:8081

   See [Published ports](#published-ports) for the full list, including the
   scheduler (8882), the Job Viewer (8885) and the supporting services.

## Lightweight Deployment (external embedding server)

If you already run an embedding server elsewhere (another host, container, or
cluster), the lightweight compose file starts the BERTrend application **plus
RabbitMQ and the scheduler** (both required by the prospective demo), but not the
embedding server. It reuses the same image built from `Dockerfile` — there is no
separate Dockerfile to maintain. The app still reserves a GPU (used by the abstractive summarizer, which loads a
torch model onto `cuda` when one is available); to run on a CPU-only host,
remove the `deploy:` block (and `CUDA_VISIBLE_DEVICES`) from the service.

```bash
EMBEDDING_SERVICE_URL=https://your-embedding-host:6464 \
BERTREND_CLIENT_SECRET=your-client-secret \
  docker compose -f docker-compose.lightweight.yml up -d
```

Alternatively, use the convenience scripts:

- `./start_bertrend_lightweight.sh` — resolves `HOST_UID`/`HOST_GID` from the
  host (so files written to mounted volumes are owned by the invoking user),
  creates the mounted host directories **including the scheduler job store**,
  checks that `EMBEDDING_SERVICE_URL` and `BERTREND_CLIENT_SECRET` are set, then
  (re)builds and starts the stack.
- `./stop_bertrend_lightweight.sh` — stops and removes the stack
  (`--volumes` also drops anonymous volumes, `--keep` only stops the
  containers). The external embedding server is never touched.

`EMBEDDING_SERVICE_URL` and `BERTREND_CLIENT_SECRET` are both **required**:
Compose refuses to start without them. Note that this applies to *every* Compose
command, `down` included — the stop script supplies placeholders for that
reason. All other variables (`OPENAI_*`, `BERTREND_BASE_DIR`, proxies, …) behave
as in the full stack and can be set in your `.env`.

> `EMBEDDING_SERVICE_USE_LOCAL` is set in the Compose files but nothing reads it.
> The real switch is `use_local` in `bertrend/config/services_default_config.toml`.
> In practice the prospective demo always calls the remote service
> (`EmbeddingService(local=False)`), and the Topic Analysis / Weak Signals demos
> let you pick local or remote in the UI.

### Scheduler job store

The scheduler resolves its APScheduler SQLite store as
`Path.home() / ".bertrend" / "db"` — it does **not** use `BERTREND_BASE_DIR`,
which is for data, logs and config. Since `HOME` is `/home/user` in the
container, Compose mounts the host side from the invoking user's home:

```yaml
- ${SCHEDULER_DB_DIR:-${HOME}/.bertrend/db}:/home/user/.bertrend/db
```

This is the same file the service used before it was containerised, so existing
scheduled jobs are picked up automatically. Set `SCHEDULER_DB_DIR` only if your
store lives somewhere else. **Stop any non-containerised scheduler first** — two
schedulers sharing one SQLite store will both claim and run the same jobs.

Use the standard `docker-compose.yml` instead when you also want the embedding
server running locally.

### Config paths in scheduled jobs

Scrape and newsletter jobs store the paths of their config files (`feed_cfg`,
`newsletter_toml_path`, `data_feed_toml_path` in the job's `json_data`). They are
stored **relative**, so that a job registered in one environment runs in another
(host and container, or two hosts sharing the job store):

- relative to `BERTREND_BASE_DIR` for user configs, e.g.
  `config/users/<user>/<model_id>_feed.toml`;
- relative to the `bertrend` package for the configs it ships, e.g.
  `bertrend_apps/config/feeds/rte_feed.toml`.

The worker resolves them against its own `BERTREND_BASE_DIR` and installed
package (`bertrend/bertrend_apps/common/portable_paths.py`).

#### Jobs registered outside Docker

Jobs registered by older versions store **absolute paths**, valid only in the
environment that registered them: `/DSIA/nlp/bertrend/config/users/...` (host
`BERTREND_BASE_DIR`) or `/home/<user>/dev/BERTrend/bertrend/bertrend_apps/config/...`
(source checkout, registered by `install_feeds_and_newsletters.py`). Since the job
store is reused as-is (see above), they reach the container, where these paths do
not exist.

The worker still runs them: an absolute path that does not exist is looked up
under the local `BERTREND_BASE_DIR` and package by its trailing components
(`config/users/...`, `bertrend_apps/config/...`), with a warning in the logs.
To rewrite them once and for all, run the migration script where `bertrend` is
installed and the scheduler is reachable. It is a dry run by default:

```bash
docker exec bertrend python -m bertrend.bertrend_apps.migrate_scheduled_job_paths
docker exec bertrend python -m bertrend.bertrend_apps.migrate_scheduled_job_paths --apply
```

- Outside the container, point it at the scheduler with
  `SCHEDULER_SERVICE_URL=http://<host>:8882/`.
- A path is rewritten when its file is found under the local `BERTREND_BASE_DIR`
  or package; otherwise it is reported and left unchanged. When the files are not
  reachable from where the script runs, pass the `BERTREND_BASE_DIR` of the
  environment that registered the jobs: `--legacy-base-dir /DSIA/nlp/bertrend`
  (repeatable).
- Job ids, triggers and next run times are kept. Job names, which embed the
  paths, are updated too by a scheduler image built from this version; older ones
  keep the old names, which is cosmetic.
- Once migrated, newsletter jobs read the configs shipped in the **installed**
  package, not the checkout's copies: keep customised configs under
  `BERTREND_BASE_DIR` instead.
- The app image installs `bertrend` from PyPI (see below): the script and the
  relative paths reach the container with the release that contains them.

Until then, two mounts make the legacy absolute paths resolve as they are:

- **Paths under the host `BERTREND_BASE_DIR`**: by default the base dir is
  mounted at `/bertrend` in the container. Set `BERTREND_CONTAINER_BASE_DIR` to
  the host path itself to mount it at the same location; the container's
  `BERTREND_BASE_DIR` follows:

  ```
  BERTREND_BASE_DIR=/DSIA/nlp/bertrend
  BERTREND_CONTAINER_BASE_DIR=/DSIA/nlp/bertrend
  ```

- **Paths into the source checkout**: Compose mounts the checkout's
  `bertrend/bertrend_apps/config` read-only at
  `${BERTREND_SOURCE_DIR}/bertrend/bertrend_apps/config`, `BERTREND_SOURCE_DIR`
  defaulting to the directory Compose is run from (`$PWD`). Run Compose from the
  checkout the jobs were registered from (`./run_compose_script.sh` does), or set
  `BERTREND_SOURCE_DIR` to that checkout's absolute path.

After changing these variables, recreate the app container (the scheduler and its
job store are left untouched):

```bash
./run_compose_script.sh up -d --no-deps --force-recreate bertrend
```

## Building the Docker Images Locally

> **The app image installs `bertrend` from PyPI** (`uv pip install -U bertrend`
> in `Dockerfile`), not from your working copy. Only `supervisord.conf` and
> `run_demos.sh` are copied from the build context, so local changes to the
> Python package do **not** reach `bertrend:latest` until they are released.
> The scheduler image is the opposite: it copies its source from the build
> context, so changes under `bertrend/services/scheduling` (including the Job
> Viewer) take effect on the next build.

A root `.dockerignore` keeps the build context small — without it the context is
several GB (`.venv`, `.git`, `data/`). It is written against what the two root
Dockerfiles actually copy, so **if you add a `COPY` to either of them, check the
path is not excluded**.

If you want to build the Docker images locally:

```bash
# Build both images
docker-compose build

# Or build individual images
docker build -t bertrend:latest -f Dockerfile .
docker build -t bertrend-embedding-server:latest -f Dockerfile.embedding_server .
```

## Running Individual Containers

### Running the Embedding Server

```bash
docker run --gpus all -p 6464:6464 \
  -v /path/to/huggingface/cache:/root/.cache/huggingface \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  -e HF_HOME=/root/.cache/huggingface \
  bertrend-embedding-server:latest
```

### Running the Main BERTrend Application

```bash
docker run --gpus all \
  -p 8083:8083 -p 8084:8084 -p 8081:8081 \
  -v /path/to/bertrend/data:/bertrend \
  -e OPENAI_API_KEY=your_key \
  -e OPENAI_BASE_URL=your_endpoint \
  -e EMBEDDING_SERVICE_URL=https://your-embedding-server:6464 \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  bertrend:latest
```

## Configuration Options

### Environment Variables

#### Main BERTrend Application

| Variable                      | Description | Default |
|-------------------------------|-------------|---------|
| `OPENAI_API_KEY`              | Your OpenAI API key — env file only, see below | - |
| `OPENAI_BASE_URL`     | OpenAI API endpoint — env file only, see below | - |
| `OPENAI_DEFAULT_MODEL`   | Default OpenAI model to use — env file only, required | - |
| `OPENAI_REASONING_EFFORT` | GPT-5 reasoning effort (`minimal`/`low`/`medium`/`high`); GPT-5 models only | `low` |
| `OPENAI_REASONING_EFFORT_TOPIC_DESCRIPTION` | Per-task override for topic description | inherits `OPENAI_REASONING_EFFORT` |
| `OPENAI_REASONING_EFFORT_SIGNAL_ANALYSIS` | Per-task override for signal analysis | inherits `OPENAI_REASONING_EFFORT` |
| `BERTREND_BASE_DIR`           | Host directory holding BERTrend data, models, logs and config, mounted into the container. Inside the container it is set to `BERTREND_CONTAINER_BASE_DIR` | `.bertrend` |
| `BERTREND_CONTAINER_BASE_DIR` | Where the host `BERTREND_BASE_DIR` is mounted in the container. Set it to the host path to keep jobs registered outside Docker working — see [Jobs registered outside Docker](#jobs-registered-outside-docker) | `/bertrend` |
| `BERTREND_SOURCE_DIR`         | Absolute path of the checkout whose `bertrend/bertrend_apps/config` is mounted read-only at the same path in the container | `$PWD` |
| `EMBEDDING_SERVICE_URL`       | URL of the embedding server (required in the lightweight stack) | `https://embedding_server:6464` |
| `EMBEDDING_SERVICE_USE_LOCAL` | Declared but unread — see the note above | `false` |
| `BERTREND_CLIENT_SECRET`      | Client secret for the embedding server (required in the lightweight stack) | - |
| `RABBITMQ_USER`               | Broker user; applied to both the broker and the workers | `guest` |
| `RABBITMQ_PASSWORD`           | Broker password; applied to both the broker and the workers | `guest` |
| `HOST_UID`                    | User ID for file permissions — resolved by the start scripts from `id -u` | `1000` |
| `HOST_GID`                    | Group ID for file permissions — resolved by the start scripts from `id -g` | `1000` |

The `OPENAI_*` variables above are passed to the app container **only from the env
file** (`BERTREND_ENV_FILE`, default `.env`), not through `environment:`. Compose
fills `${VAR}` in the compose file from the invoking shell before `.env`, so a
shell exporting `OPENAI_BASE_URL` for another tool (e.g. a direct Azure endpoint)
would otherwise override the env file and send BERTrend's LLM calls there. Check
what the container got with `docker exec bertrend printenv OPENAI_BASE_URL OPENAI_DEFAULT_MODEL`.

#### Scheduler and Job Viewer

| Variable | Description | Default |
|----------|-------------|---------|
| `SCHEDULER_DB_DIR` | Host directory holding the APScheduler SQLite store | `$HOME/.bertrend/db` |
| `SCHEDULER_JOBS_URL` | Jobs endpoint the Job Viewer polls | `http://scheduler:8000/jobs` in Compose |
| `MAX_WORKERS` | Maximum concurrent scheduled jobs | `100` |

#### Embedding Server

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_RATE_LIMIT` | Rate limit for API requests | `50` |
| `DEFAULT_RATE_WINDOW` | Time window for rate limiting (seconds) | `60` |
| `HF_HOME` | Hugging Face cache directory | `/root/.cache/huggingface` |
| `HOST_UID` | User ID for file permissions | `1000` |
| `HOST_GID` | Group ID for file permissions | `1000` |

### Volume Mounts

#### Main BERTrend Application

Mount a directory to `/bertrend` to persist data:

```bash
-v /path/on/host:/bertrend
```

To mount it elsewhere, mount it at that path and set `BERTREND_BASE_DIR` to match
(Compose does both from `BERTREND_CONTAINER_BASE_DIR`):

```bash
-v /path/on/host:/path/on/host -e BERTREND_BASE_DIR=/path/on/host/
```

#### Embedding Server

Mount a directory to the Hugging Face cache to avoid re-downloading models:

```bash
-v /path/to/huggingface/cache:/root/.cache/huggingface
```

## GPU Support

Both containers support GPU acceleration. The embedding server is the main
consumer; in the app container the GPU is used by the abstractive summarizer,
which moves its model to `cuda` when one is available. Note that the topic
modelling stack itself is CPU-only here — `umap-learn` and `hdbscan` are the
CPU implementations, not the cuML ones.

1. Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

2. The Compose files (`docker-compose.yml` and `docker-compose.lightweight.yml`)
   enable GPU by default via their `deploy` sections. To run on a CPU-only host,
   remove those `deploy` sections (and `CUDA_VISIBLE_DEVICES`).

3. When running containers individually, add the `--gpus all` flag.

## Automated Docker Image Publishing

BERTrend uses GitHub Actions to automatically build and publish Docker images to Docker Hub when changes are pushed to the main branch. The workflow is defined in `.github/workflows/docker-publish.yml`.

The images are published to:
- `rte-france/bertrend:latest`
- `rte-france/bertrend-embedding-server:latest`

## Troubleshooting

### Common Issues

1. **Permission Issues**: If you encounter permission problems with mounted volumes, ensure the `HOST_UID` and `HOST_GID` environment variables match your user's UID and GID.

2. **GPU Not Detected**: Verify that the NVIDIA Container Toolkit is properly installed and that your GPU drivers are up to date.

3. **Embedding Server Connection Failure**: Check that the embedding server is running and that the `EMBEDDING_SERVICE_URL` is correctly set in the main application. A 401 on `/token` means `BERTREND_CLIENT_SECRET` does not match the secret registered for the `bertrend` client on that server.

4. **Scheduler unhealthy, `sqlite3.OperationalError: unable to open database file`**:
   the job-store directory does not exist on the host, so Docker created the
   bind-mount target as `root` and the container — running as `HOST_UID` —
   cannot write to it. The start script creates it for you; to repair an
   existing one:

   ```bash
   sudo chown -R $(id -u):$(id -g) ~/.bertrend/db
   ```

   The app then fails too, with `dependency failed to start`, because it waits
   on `scheduler: service_healthy`.

5. **Nothing answers on any port**: check whether the port mappings carry a
   `127.0.0.1:` prefix. That binds them to the server's loopback interface, so
   they answer to `curl http://localhost:8083` *on the server* but to nothing
   from another machine. The Compose files publish on all interfaces.

6. **Job Viewer (8885) loads but shows no jobs**: it polls `SCHEDULER_JOBS_URL`.
   Under Compose this is `http://scheduler:8000/jobs`; run outside Compose it
   falls back to a hardcoded public URL. Check the value with
   `docker exec bertrend-job-viewer env | grep SCHEDULER_JOBS_URL`.

7. **Queue jobs fail with `FileNotFoundError` on a host path**
   (`/DSIA/...`, `/home/<user>/dev/BERTrend/...`): the job was registered by an
   older version, outside Docker, and stores an absolute host path. Migrate it, or
   mount the paths: see
   [Jobs registered outside Docker](#jobs-registered-outside-docker).

8. **Scrape jobs fail with `PermissionError: [Errno 13]` on a `.jsonl` under
   `data/feeds`**: the scraper appends to the day's file, which was created by
   another UID than the container's `HOST_UID` — typically the pre-Docker setup
   running under another account, or a container that ran as root. List the files
   the container user does not own, then take them back:

   ```bash
   find "$BERTREND_BASE_DIR" ! -uid $(id -u) -printf '%u:%g %p\n' | head
   sudo chown -R $(id -u):$(id -g) "$BERTREND_BASE_DIR"
   ```

   Also check that nothing from the old setup still runs on the host
   (`crontab -l`, `ps aux | grep -E 'bertrend_worker|scheduler'`), or it will keep
   creating files with the wrong owner.

### Logs

To view container logs:

```bash
# View logs for all containers
docker-compose logs

# View logs for a specific container
docker-compose logs bertrend
docker-compose logs embedding_server
docker-compose logs scheduler
docker-compose logs job_viewer

# Follow logs in real-time
docker-compose logs -f
```