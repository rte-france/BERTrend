import os
from dataclasses import dataclass


@dataclass
class RabbitMQConfig:
    """RabbitMQ configuration"""

    host: str = os.getenv("RABBITMQ_HOST", "localhost")
    port: int = int(os.getenv("RABBITMQ_PORT", 5672))
    username: str = os.getenv("RABBITMQ_USER", "guest")
    password: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    virtual_host: str = os.getenv("RABBITMQ_VHOST", "/")

    # Queue configuration
    request_queue: str = "bertrend_requests"
    response_queue: str = "bertrend_responses"
    error_queue: str = "bertrend_failed"

    # Performance tuning
    prefetch_count: int = int(os.getenv("RABBITMQ_PREFETCH_COUNT", 1))

    # Retry configuration
    max_retries: int = int(os.getenv("RABBITMQ_MAX_RETRIES", 3))
    retry_delay: int = int(os.getenv("RABBITMQ_RETRY_DELAY", 5000))  # milliseconds
