#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import time
from urllib.parse import urlparse

import requests
from loguru import logger

from bertrend.services.queue.queue_manager import QueueManager
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig


# ====================================
# Job Functions (must be module-level for ProcessPoolExecutor)
# ====================================
def sample_job(message: str = "Default message"):
    """Example job function - must be at module level for multiprocessing"""
    logger.info(f"Executing job: {message}")
    time.sleep(1)  # Simulate some work
    return f"Completed: {message}"


def http_request(
    url: str,
    method: str = "POST",
    json_data: dict = None,
    timeout: int = 600,  # 10 min
):
    """Send request via RabbitMQ queue instead of direct HTTP"""
    # Extract endpoint from URL (e.g., "http://host:port/scrape-feed" → "/scrape-feed")
    endpoint = urlparse(url).path

    config = RabbitMQConfig()
    queue_manager = QueueManager(config)
    queue_manager.connect()

    request_data = {
        "endpoint": endpoint,
        "method": method,
        "json_data": json_data or {},
    }

    correlation_id = queue_manager.publish_request(request_data)
    queue_manager.close()

    return {"status": "queued", "correlation_id": correlation_id}


def http_request_old(
    url: str,
    method: str = "GET",
    headers: dict = None,
    json_data: dict = None,
    timeout: int = 600,  # 10 min
):
    """Execute an HTTP request (curl-like functionality)"""
    logger.info(f"Executing HTTP {method} request to {url}")
    try:
        with requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=json_data,
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                response.raise_for_status()
            logger.info(
                f"HTTP request completed with status code: {response.status_code}"
            )
            return f"Request to {url} completed with status {response.status_code}"
    except Exception as e:
        logger.error(f"HTTP request failed: {str(e)}")
        raise RuntimeError(f"Error executing request: {str(e)}")


# Job function registry
JOB_FUNCTIONS = {
    "http_request": http_request,
    "sample_job": sample_job,
}
