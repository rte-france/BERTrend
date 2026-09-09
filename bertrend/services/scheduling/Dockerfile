FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/bertrend/services/scheduling

# Give the container user a name: the scheduler and job_viewer also run as an
# arbitrary numeric HOST_UID with no /etc/passwd entry, which makes
# getpass.getuser() raise "OSError: No username set in the environment".
# Nothing here imports torch (the caller that tripped over it in the app image),
# so this is precautionary -- kept identical to the other two Dockerfiles.
ENV USER=bertrend \
    LOGNAME=bertrend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Run the application
CMD ["uvicorn", "bertrend.services.scheduling.scheduling_service:app", "--host", "0.0.0.0", "--port", "8000"]
