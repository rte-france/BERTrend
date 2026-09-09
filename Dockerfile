FROM python:3.13-slim-bookworm
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
    cron \
    locales \
    && echo "fr_FR.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen fr_FR.UTF-8 \
    && update-locale LANG=fr_FR.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally to /usr/local/bin
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set workdir
WORKDIR /app

# Use ARG to allow build-time variables
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG BERTREND_BASE_DIR=/bertrend/

# Ensure app directory has appropriate permissions. /.streamlit is the container
# user's HOME/.streamlit -- where the secrets file is bind-mounted at runtime.
# NB. no /app/nltk_data: NLTK refuses to use a world- or group-writable data
# directory, so the corpora are baked into a root-owned one further down.
RUN chmod -R 777 /app && \
    mkdir -p /.streamlit && \
    chmod -R 777 /.streamlit

COPY supervisord.conf run_demos.sh /app/

# Install BERTrend from THIS checkout, not from PyPI.
#
# The image used to do `uv pip install -U bertrend`, which meant a deployment ran
# whatever was last published to PyPI rather than the branch being deployed --
# every un-released change was silently absent from the running container.
#
# Split in two layers so an ordinary code change does not reinstall torch & co:
#   1. dependencies, pinned from uv.lock -- only invalidated by pyproject/uv.lock
#   2. the project itself, installed with --no-deps
# Staged in /src rather than /app on purpose: run_demos.sh resolves BERTREND_HOME
# via `python -c "import bertrend"` with cwd=/app, and for `python -c` sys.path[0]
# is the cwd -- a source copy at /app/bertrend would shadow the installed package
# and point every supervisord program at the build material instead.
COPY pyproject.toml uv.lock README.md LICENSE.md AUTHORS.txt /src/
RUN uv export --project /src --frozen --no-dev --no-emit-project --no-hashes \
        --format requirements-txt -o /tmp/requirements.txt && \
    uv pip install --no-cache-dir --system -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY bertrend /src/bertrend
RUN uv pip install --no-cache-dir --system --no-deps /src && \
    chmod -R a+w /usr/local/lib/python3.13/site-packages/ # Workaround for packages (such as numba which use caching in __pycache__ (requires writing rights)

# Pre-download the NLTK corpora the code needs: `stopwords` (bertrend.utils, on
# every service start) and `punkt`/`punkt_tab` (the extractive summarizer).
# Baking them in fixes two runtime failures seen in the container:
#   - "NLTK will not authorize the non-private download directory
#     '/app/nltk_data': it (or an ancestor) is world- or group-writable" -- that
#     dir is chmod 777 because the container runs as an arbitrary HOST_UID, and
#     NLTK refuses to use a world-writable data dir at all.
#   - "refusing a proxied fetch ... SSRF protection cannot be enforced
#     (CWE-918)" -- egress goes through an HTTP proxy, so NLTK cannot pin the
#     validated IP. NLTK_ALLOW_PROXIED_URLOPEN=1 opts into that here at build
#     time, where the proxy is the trusted corporate one.
# The target is root-owned and read-only, which is exactly what NLTK wants.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "\
import nltk; \
[nltk.download(pkg, download_dir='/usr/local/share/nltk_data', raise_on_error=True) \
 for pkg in ('stopwords', 'punkt', 'punkt_tab')]" && \
    chmod -R a-w,a+rX /usr/local/share/nltk_data

# Give the container user a name. It runs as an arbitrary numeric HOST_UID that
# has no /etc/passwd entry, so getpass.getuser() -- which checks LOGNAME/USER
# first and only then falls back to pwd.getpwuid(os.getuid()) -- raised
# "OSError: No username set in the environment". torch calls it at import time
# to name its inductor cache dir, so *any* module importing torch died:
#   torch/_inductor/runtime/cache_dir_utils.py -> default_cache_dir()
#     -> getpass.getuser()
# which took down the prospective demo (8081) via bertopic -> sentence_transformers.
# The value is only used to build cache paths; it does not change the runtime uid.
ENV USER=bertrend \
    LOGNAME=bertrend

# Expose the Streamlit demos and the FastAPI services
EXPOSE 8081 8083 8084 8091 8881 8886 8887

# Set the entrypoint
ENTRYPOINT ["/app/run_demos.sh"]

# To run this container with GPU support, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -e OPENAI_API_KEY=your_key -e OPENAI_BASE_URL=your_endpoint bertrend:latest
#
# To mount a host directory to BERTREND_BASE_DIR, use:
# docker run --gpus all -p 8501:8501 -p 8502:8502 -p 8503:8503 -v /path/on/host:/bertrend/ bertrend:latest
#
# Access the demos at:
# - Topic Analysis: http://localhost:8083
# - Weak Signals: http://localhost:8084
# - Prospective Demo: http://localhost:8081