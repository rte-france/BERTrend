#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import time

import requests
from fastapi import HTTPException
from loguru import logger


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
    method: str = "GET",
    headers: dict = None,
    json_data: dict = None,
    timeout: int = 30,
):
    """Execute an HTTP request (curl-like functionality)"""
    logger.info(f"Executing HTTP {method} request to {url}")
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=json_data,
            timeout=timeout,
        )
        if response.status_code != 200:
            response.raise_for_status()
        logger.info(f"HTTP request completed with status code: {response.status_code}")
        return f"Request to {url} completed with status {response.status_code}"
    except Exception as e:
        logger.error(f"HTTP request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error executing request: {str(e)}"
        )


# Job function registry
JOB_FUNCTIONS = {
    "http_request": http_request,
    "sample_job": sample_job,
}
