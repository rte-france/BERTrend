# Running BERTrend with Docker

This guide explains how to use BERTrend with Docker, which provides an easy way to run the application without installing dependencies directly on your system.

## Docker Images Overview

BERTrend provides two Docker images:

1. **Main BERTrend Image** (`bertrend:latest`): Contains the core BERTrend application and three demo applications:
   - Topic Analysis Demo (port 8083)
   - Weak Signals Demo (port 8084)
   - Prospective Demo (port 8081)

2. **Embedding Server Image** (`bertrend-embedding-server:latest`): Provides embedding services for the main application, running on port 6464.

Both images are built with NVIDIA CUDA support for GPU acceleration.

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

## Lightweight Deployment (external embedding server)

If you already run an embedding server elsewhere (another host, container, or
cluster), you can start **only the main BERTrend application** with the
lightweight compose file. It does not build or start the embedding server, and
reuses the same image built from `Dockerfile` — there is no separate Dockerfile
to maintain. The app still reserves a GPU (used to speed up topic modelling —
BERTopic / UMAP / HDBSCAN); to run on a CPU-only host, remove the `deploy:`
block (and `CUDA_VISIBLE_DEVICES`) from the service.

```bash
EMBEDDING_SERVICE_URL=https://your-embedding-host:6464 \
  docker compose -f docker-compose.lightweight.yml up -d
```

Alternatively, use the convenience script `./start_bertrend_lightweight.sh`,
which sets `HOST_UID`/`HOST_GID` (so files written to mounted volumes are owned
by the host user), creates the mounted host directories, checks that
`EMBEDDING_SERVICE_URL` is set, and (re)builds and starts the stack.

`EMBEDDING_SERVICE_URL` is **required** (compose refuses to start without it) and
must point to your running embedding server; `EMBEDDING_SERVICE_USE_LOCAL` is
forced to `false`. All other variables (`OPENAI_*`, `BERTREND_BASE_DIR`,
`BERTREND_CLIENT_SECRET`, proxies, …) behave as in the full stack and can be set
in your `.env`.

Use the standard `docker-compose.yml` instead when you also want the embedding
server running locally.

## Building the Docker Images Locally

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
| `OPENAI_API_KEY`              | Your OpenAI API key | - |
| `OPENAI_BASE_URL`     | OpenAI API endpoint | - |
| `OPENAI_DEFAULT_MODEL`   | Default OpenAI model to use | `gpt-5.6-luna` |
| `OPENAI_REASONING_EFFORT` | GPT-5 reasoning effort (`minimal`/`low`/`medium`/`high`); GPT-5 models only | `low` |
| `OPENAI_REASONING_EFFORT_TOPIC_DESCRIPTION` | Per-task override for topic description | inherits `OPENAI_REASONING_EFFORT` |
| `OPENAI_REASONING_EFFORT_SIGNAL_ANALYSIS` | Per-task override for signal analysis | inherits `OPENAI_REASONING_EFFORT` |
| `BERTREND_BASE_DIR`           | Base directory for BERTrend data | `/bertrend/` |
| `EMBEDDING_SERVICE_URL`       | URL of the embedding server | `https://embedding_server:6464` |
| `EMBEDDING_SERVICE_USE_LOCAL` | Whether to use local embeddings | `false` |
| `HOST_UID`                    | User ID for file permissions | `1000` |
| `HOST_GID`                    | Group ID for file permissions | `1000` |

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

#### Embedding Server

Mount a directory to the Hugging Face cache to avoid re-downloading models:

```bash
-v /path/to/huggingface/cache:/root/.cache/huggingface
```

## GPU Support

Both containers support GPU acceleration, used both by the embedding server and
to speed up topic modelling (BERTopic / UMAP / HDBSCAN) in the main application.

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

3. **Embedding Server Connection Failure**: Check that the embedding server is running and that the `EMBEDDING_SERVICE_URL` is correctly set in the main application.

### Logs

To view container logs:

```bash
# View logs for all containers
docker-compose logs

# View logs for a specific container
docker-compose logs bertrend
docker-compose logs embedding_server

# Follow logs in real-time
docker-compose logs -f
```