FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    gfortran \
    less \
    apt-transport-https \
    tar \
    wget \
    curl \
    sudo \
    locales \
    && echo "fr_FR.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen fr_FR.UTF-8 \
    && update-locale LANG=fr_FR.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally to /usr/local/bin
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use ARG to allow build-time variables
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG BERTREND_BASE_DIR=/bertrend/

# Create NLTK data directory and ensure app directory has appropriate permissions
RUN mkdir -p /app/nltk_data && \
    chmod -R 777 /app/nltk_data && \
    chmod -R 777 /app

COPY supervisord.conf run_demos.sh /app/

# Install BERTrend
RUN uv pip install --no-cache-dir --system -U bertrend[apps]

# Set workdir
WORKDIR /app

# Expose Streamlit ports for all three demos
EXPOSE 8081 8083 8084

# Set the entrypoint
ENTRYPOINT ["/app/run_demos.sh"]

# To run this container with GPU support, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -e OPENAI_API_KEY=your_key -e OPENAI_ENDPOINT=your_endpoint bertrend:latest
#
# To mount a host directory to BERTREND_BASE_DIR, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -v /path/on/host:/bertrend/ bertrend:latest
#
# Access the demos at:
# - Topic Analysis: http://localhost:8083
# - Weak Signals: http://localhost:8084
# - Prospective Demo: http://localhost:8081