#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Pytest tests to verify that job_functions.http_request properly raises
RuntimeError instead of HTTPException, which fixes the pickling issue
in multiprocessing contexts.
"""

import pickle
import pytest
from concurrent.futures import ProcessPoolExecutor

from bertrend.services.scheduling.job_utils.job_functions import http_request


def job_wrapper_for_pool():
    """Module-level wrapper to call http_request in a subprocess"""
    try:
        http_request("http://invalid-host-that-does-not-exist-12345.com", timeout=1)
    except RuntimeError as e:
        return f"RuntimeError: {str(e)[:50]}"
    except Exception as e:
        return f"OtherError: {type(e).__name__}: {str(e)[:50]}"


def test_exception_is_picklable():
    """Test that the exception raised by http_request can be pickled"""
    with pytest.raises(RuntimeError) as exc_info:
        # Try to trigger an exception by connecting to an invalid URL
        http_request("http://invalid-host-that-does-not-exist-12345.com", timeout=1)

    # Verify the exception was caught
    exception = exc_info.value
    assert isinstance(exception, RuntimeError)

    # Try to pickle the exception
    pickled = pickle.dumps(exception)
    unpickled = pickle.loads(pickled)

    # Verify unpickling worked correctly
    assert isinstance(unpickled, RuntimeError)
    assert type(exception).__name__ == type(unpickled).__name__


def test_exception_in_process_pool():
    """Test that http_request exceptions work properly in ProcessPoolExecutor"""
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(job_wrapper_for_pool)
        result = future.result(timeout=10)

        # Verify that the RuntimeError was properly handled across process boundary
        assert "RuntimeError" in result
        assert "Error executing request" in result


def test_timeout_scenario():
    """Test with a timeout scenario (simulates the original error)"""
    with pytest.raises(RuntimeError) as exc_info:
        # Use a valid host but very short timeout to trigger timeout
        http_request("https://www.google.com", timeout=0.001)

    error_msg = str(exc_info.value)

    # Verify it's a timeout-related error
    assert "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
    assert "Error executing request" in error_msg
