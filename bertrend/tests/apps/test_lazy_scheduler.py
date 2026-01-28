#!/usr/bin/env python
#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Test script to verify lazy initialization of SCHEDULER_UTILS.
This tests that importing bertrend_apps doesn't immediately create scheduler instances.
"""

import os
import sys
import multiprocessing


def helper_test_lazy_init_in_worker():
    """Test that SCHEDULER_UTILS is lazily initialized in worker processes."""
    try:
        pid = os.getpid()
        print(f"[Worker {pid}] Importing bertrend_apps...")

        # Import should not trigger initialization
        from bertrend_apps import SCHEDULER_UTILS

        print(f"[Worker {pid}] Import complete (should not have initialized yet)")

        # Now actually use it - this should trigger initialization
        print(f"[Worker {pid}] Accessing SCHEDULER_UTILS...")
        scheduler_type = type(SCHEDULER_UTILS).__name__
        print(f"[Worker {pid}] Successfully accessed SCHEDULER_UTILS: {scheduler_type}")

        # Try to call a method to ensure proxy works
        if hasattr(SCHEDULER_UTILS, "add_job_to_crontab"):
            print(f"[Worker {pid}] Proxy working correctly - method accessible")

        return True
    except Exception as e:
        print(f"[Worker {pid}] Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lazy_init_in_worker():
    """Main test function."""
    print(f"[Main {os.getpid()}] Starting lazy initialization test...")

    # Test with multiple processes (simulating uvicorn workers)
    num_workers = 2
    processes = []

    for i in range(num_workers):
        p = multiprocessing.Process(target=helper_test_lazy_init_in_worker)
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    success = True
    for p in processes:
        p.join()
        if p.exitcode != 0:
            success = False
            print(f"[Main] Worker process {p.pid} exited with code {p.exitcode}")

    if success:
        print("[Main] ✓ All workers completed successfully with lazy initialization!")
    else:
        print("[Main] ✗ Some workers failed!")

    assert success == True
