#!/usr/bin/env python
#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Test script to verify the multi-worker fix.
This script simulates what happens when uvicorn forks worker processes.
"""

import os
import sys
import multiprocessing


def helper_test_import_in_worker():
    """Test that importing bertrend_apps works in a forked worker process."""
    try:
        # Import bertrend_apps which triggers SCHEDULER_UTILS initialization
        from bertrend_apps import SCHEDULER_UTILS

        print(
            f"[Worker {os.getpid()}] Successfully imported SCHEDULER_UTILS: {type(SCHEDULER_UTILS).__name__}"
        )
        return True
    except Exception as e:
        print(f"[Worker {os.getpid()}] Failed to import: {e}")
        return False


def test_multiworker():
    """Main test function."""
    print(f"[Main {os.getpid()}] Starting multi-worker import test...")

    # Test with multiple processes (simulating uvicorn workers)
    num_workers = 2
    processes = []

    for i in range(num_workers):
        p = multiprocessing.Process(target=helper_test_import_in_worker)
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
        print("[Main] ✓ All workers completed successfully!")
    else:
        print("[Main] ✗ Some workers failed!")

    assert success
