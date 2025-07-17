FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    gfortran \
    less \
    apt-transport-https \
    git \
    ssh \
    tar \
    wget \
    curl \
    sudo \
    locales \
    && locale-gen fr_FR.UTF-8 \
    && update-locale LANG=fr_FR.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Set the locale environment variable
ENV LANG=fr_FR.UTF-8
ENV LC_TIME=fr_FR.UTF-8
ENV LC_ALL=fr_FR.UTF-8

# Install gosu for user switching
RUN wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/1.17/gosu-$(dpkg --print-architecture)" && \
    chmod +x /usr/local/bin/gosu

# Install uv globally to /usr/local/bin
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv

# Use ARG to allow build-time variables
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HF_HOME

# Create user and group first
RUN groupadd -g $HOST_GID hostgroup || true
RUN useradd -u $HOST_UID -g $HOST_GID -m -s /bin/bash hostuser || true
RUN usermod -aG sudo hostuser || true
RUN echo "hostuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Setup Python venv as the host user
RUN mkdir /opt/venv
RUN chown -R $HOST_UID:$HOST_GID /opt/venv
RUN gosu $HOST_UID:$HOST_GID uv python install 3.12
RUN gosu $HOST_UID:$HOST_GID uv venv /opt/venv --python 3.12

# Create python symlink if it doesn't exist
RUN if [ ! -f /opt/venv/bin/python ]; then ln -s /opt/venv/bin/python3 /opt/venv/bin/python; fi

# Ensure the virtual environment's bin directory is in the PATH
ENV PATH=/opt/venv/bin:$PATH

# Install BERTrend
RUN gosu $HOST_UID:$HOST_GID uv pip install -U bertrend[apps]

ARG BERTREND_BASE_DIR=/bertrend/

# Create necessary directories
RUN mkdir -p /bertrend \
    $BERTREND_BASE_DIR/data \
    $BERTREND_BASE_DIR/cache \
    $BERTREND_BASE_DIR/output \
    $BERTREND_BASE_DIR/config \
    $BERTREND_BASE_DIR/logs/bertrend

RUN chown -R $HOST_UID:$HOST_GID /bertrend $BERTREND_BASE_DIR

# Set working directory
WORKDIR /

# Create the startup script with proper variable handling
RUN BERTREND_HOME=$(python -c "import os; import bertrend; print(os.path.dirname(os.path.dirname(bertrend.__file__)))") && \
    printf '#!/bin/bash\n\
# Set BERTREND_HOME from the captured value\n\
export BERTREND_HOME=%s\n\
export BERTREND_BASE_DIR=%s\n\
\n\
echo "Starting BERTrend demos..."\n\
echo "BERTREND_HOME: $BERTREND_HOME"\n\
echo "BERTREND_BASE_DIR: $BERTREND_BASE_DIR"\n\
\n\
source /opt/venv/bin/activate\n\
# Start Topic Analysis demo on port 8501\n\
cd $BERTREND_HOME/bertrend/demos/topic_analysis && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 2>&1 | tee -a $BERTREND_BASE_DIR/logs/bertrend/topic_analysis_demo.log &\n\
\n\
# Start Weak Signals demo on port 8502\n\
cd $BERTREND_HOME/bertrend/demos/weak_signals && streamlit run app.py --server.port=8502 --server.address=0.0.0.0 2>&1 | tee -a $BERTREND_BASE_DIR/logs/bertrend/weak_signals_demo.log &\n\
\n\
# Start Prospective demo on port 8503\n\
cd $BERTREND_HOME/bertrend_apps/prospective_demo && streamlit run app.py --server.port=8503 --server.address=0.0.0.0 2>&1 | tee -a $BERTREND_BASE_DIR/logs/bertrend/prospective_analysis_demo.log &\n\
\n\
# Keep the container running\n\
wait\n\
' "$BERTREND_HOME" "$BERTREND_BASE_DIR" > start_demo.sh && chmod +x start_demo.sh

ENV NVIDIA_VISIBLE_DEVICES=all

# Expose Streamlit ports for all three demos
EXPOSE 8501 8502 8503

# Set the entrypoint
ENTRYPOINT ["/start_demo.sh"]

# To run this container with GPU support, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -e OPENAI_API_KEY=your_key -e OPENAI_ENDPOINT=your_endpoint bertrend:latest
#
# To mount a host directory to BERTREND_BASE_DIR, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -v /path/on/host:/bertrend/ bertrend:latest
#
# Access the demos at:
# - Topic Analysis: http://localhost:8501
# - Weak Signals: http://localhost:8502
# - Prospective Demo: http://localhost:8503