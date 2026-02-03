#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Pytest tests to verify exception handling and pickling for job functions.
"""

import pickle
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import pytest

from bertrend.services.scheduling.job_utils.job_functions import (
    basic_http_request,
    http_request,
)


def job_wrapper_old():
    """Module-level wrapper that calls http_request_old"""
    try:
        basic_http_request(
            "http://invalid-host-that-does-not-exist-12345.com", timeout=1
        )
    except RuntimeError as e:
        return f"RuntimeError: {str(e)[:50]}"
    except Exception as e:
        return f"OtherError: {type(e).__name__}: {str(e)[:50]}"


def test_exception_is_picklable():
    """Test that the exception raised by http_request_old can be pickled"""
    with pytest.raises(RuntimeError) as exc_info:
        # Try to trigger an exception by connecting to an invalid URL
        basic_http_request(
            "http://invalid-host-that-does-not-exist-12345.com", timeout=1
        )

    # Verify the exception was caught
    exception = exc_info.value
    assert isinstance(exception, RuntimeError)

    # Try to pickle the exception
    pickled = pickle.dumps(exception)
    unpickled = pickle.loads(pickled)

    # Verify unpickling worked correctly
    assert isinstance(unpickled, RuntimeError)
    assert type(exception).__name__ == type(unpickled).__name__


def test_http_request_returns_queued_status():
    """Test that the new http_request returns a queued status"""
    with patch(
        "bertrend.services.scheduling.job_utils.job_functions.QueueManager"
    ) as mock_qm:
        mock_qm.return_value.publish_request.return_value = "test-corr-id"

        result = http_request("http://localhost/test")

        assert result["status"] == "queued"
        assert result["correlation_id"] == "test-corr-id"


def test_exception_in_process_pool():
    """Test that http_request_old exceptions work properly in ProcessPoolExecutor"""
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(job_wrapper_old)
        result = future.result(timeout=10)

        # Verify that the RuntimeError was properly handled across process boundary
        assert "RuntimeError" in result
        assert "Error executing request" in result


def test_timeout_scenario():
    """Test with a timeout scenario using http_request_old"""
    with pytest.raises(RuntimeError) as exc_info:
        # Use a valid host but very short timeout to trigger timeout
        basic_http_request("https://www.google.com", timeout=0.001)

    error_msg = str(exc_info.value)

    # Verify it's a timeout-related or network-related error
    assert (
        "timeout" in error_msg.lower()
        or "timed out" in error_msg.lower()
        or "network is unreachable" in error_msg.lower()
    )
    assert "Error executing request" in error_msg
