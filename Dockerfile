FROM 2.7.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND noninteractive


# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3.8 python3.8-dev python3-pip\
    gfortran \
    less \
    apt-transport-https \
    git\
    ssh \
    tar

# Create and activate conda environment
RUN conda create python=3.12 --name bertrend
SHELL ["bertrend", "run", "-n", "lips", "/bin/bash", "-c"]


# Create necessary directories
RUN mkdir -p $BERTREND_BASE_DIR/data \
    $BERTREND_BASE_DIR/cache \
    $BERTREND_BASE_DIR/output \
    $BERTREND_BASE_DIR/config \
    $BERTREND_BASE_DIR/logs/bertrend

# Define volume for BERTREND_BASE_DIR to allow sharing with host OS
VOLUME $BERTREND_BASE_DIR

RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install project dependencies
# Note: We're using Python 3.10 instead of 3.12, so we need to override the Python version requirement
RUN uv pip install bertrend[apps]

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
