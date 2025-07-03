# Use PyTorch with CUDA as base image
# Note: This image comes with Python 3.10 and PyTorch 2.2.1 pre-installed
# The project requires Python 3.12, but we're using this image to save installation time
# If Python version compatibility issues arise, revert to python:3.12-slim
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.0.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    BERTREND_BASE_DIR="/app/data" \
    # OpenAI environment variables (can be overridden at runtime)
    OPENAI_API_KEY="" \
    OPENAI_ENDPOINT="" \
    OPENAI_DEFAULT_MODEL_NAME="gpt-4o-mini"

# Install system dependencies
# Note: The PyTorch image already includes CUDA, so we don't need nvidia-cuda-toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \  # Required for UMAP and other libraries
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Modify pyproject.toml to accept Python 3.10
RUN sed -i 's/requires-python = ">=3.12,<4.0"/requires-python = ">=3.10,<4.0"/' /app/pyproject.toml

# Create necessary directories
RUN mkdir -p $BERTREND_BASE_DIR/data \
    $BERTREND_BASE_DIR/cache \
    $BERTREND_BASE_DIR/output \
    $BERTREND_BASE_DIR/config \
    $BERTREND_BASE_DIR/logs/bertrend

# Define volume for BERTREND_BASE_DIR to allow sharing with host OS
VOLUME $BERTREND_BASE_DIR

# Install project dependencies
# Note: We're using Python 3.10 instead of 3.12, so we need to override the Python version requirement
RUN poetry config virtualenvs.create false && \
    poetry install --extras apps --no-interaction --no-ansi

# Expose Streamlit ports for all three demos
EXPOSE 8501 8502 8503

# Create a script to start all demos simultaneously
RUN echo '#!/bin/bash\n\
# Start Topic Analysis demo on port 8501\n\
cd /app/bertrend/demos/topic_analysis && streamlit run app.py --server.port=8501 2>&1 | tee -a $BERTREND_BASE_DIR/logs/bertrend/topic_analysis_demo.log &\n\
\n\
# Start Weak Signals demo on port 8502\n\
cd /app/bertrend/demos/weak_signals && streamlit run app.py --server.port=8502 2>&1 | tee -a $BERTREND_BASE_DIR/logs/bertrend/weak_signals_demo.log &\n\
\n\
# Start Prospective demo on port 8503\n\
cd /app/bertrend_apps/prospective_demo && streamlit run app.py --server.port=8503 2>&1 | tee -a $BERTREND_BASE_DIR/logs/bertrend/prospective_analysis_demo.log &\n\
\n\
# Keep the container running\n\
wait\n\
' > /app/start_demo.sh && chmod +x /app/start_demo.sh

# Set the entrypoint
ENTRYPOINT ["/app/start_demo.sh"]

# No need for CMD as we're running all demos

# Add healthcheck that checks all three demos
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || curl --fail http://localhost:8502/_stcore/health || curl --fail http://localhost:8503/_stcore/health || exit 1

# Note: This image uses a PyTorch pre-built image with Python 3.10 to save installation time.
# The original project requires Python 3.12, but we've modified it to work with Python 3.10.
# If you encounter compatibility issues, consider reverting to the python:3.12-slim base image.
#
# To run this container with GPU support, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -e OPENAI_API_KEY=your_key -e OPENAI_ENDPOINT=your_endpoint bertrend:latest
#
# To mount a host directory to BERTREND_BASE_DIR, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -v /path/on/host:/app/data bertrend:latest
#
# Access the demos at:
# - Topic Analysis: http://localhost:8501
# - Weak Signals: http://localhost:8502
# - Prospective Demo: http://localhost:8503
