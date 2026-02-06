#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
"""
Stress test for session management and file descriptor monitoring.
Tests the APSchedulerUtils HTTP session management under load.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import Mock, patch

import psutil
import pytest

from bertrend.bertrend_apps.common.apscheduler_utils import APSchedulerUtils, _request

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))


class FileDescriptorMonitor:
    """Monitor file descriptor usage during tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_fds = None
        self.peak_fds = 0
        self.monitoring = False
        self.fd_history = []

    def start(self):
        """Start monitoring file descriptors."""
        self.initial_fds = self.get_fd_count()
        self.peak_fds = self.initial_fds
        self.monitoring = True
        self.fd_history = [(time.time(), self.initial_fds)]
        print(f"📊 Initial file descriptors: {self.initial_fds}")

    def get_fd_count(self):
        """Get current number of open file descriptors."""
        try:
            return self.process.num_fds()  # Linux/Mac
        except AttributeError:
            # Windows fallback
            return len(self.process.open_files()) + len(self.process.connections())

    def update(self):
        """Update monitoring data."""
        if not self.monitoring:
            return
        current_fds = self.get_fd_count()
        self.fd_history.append((time.time(), current_fds))
        if current_fds > self.peak_fds:
            self.peak_fds = current_fds
        return current_fds

    def stop(self):
        """Stop monitoring and return results."""
        self.monitoring = False
        final_fds = self.get_fd_count()
        leaked_fds = final_fds - self.initial_fds

        print(f"\n{'=' * 60}")
        print("📊 File Descriptor Report")
        print(f"{'=' * 60}")
        print(f"Initial FDs:  {self.initial_fds}")
        print(f"Peak FDs:     {self.peak_fds} (+{self.peak_fds - self.initial_fds})")
        print(f"Final FDs:    {final_fds}")
        print(f"Leaked FDs:   {leaked_fds}")
        print(f"{'=' * 60}\n")

        return {
            "initial": self.initial_fds,
            "peak": self.peak_fds,
            "final": final_fds,
            "leaked": leaked_fds,
            "history": self.fd_history,
        }


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code=200, json_data=None, text="OK"):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.content = text.encode("utf-8")

    def json(self):
        return self._json_data


def mock_request_success(*args, **kwargs):
    """Mock successful HTTP request."""
    return MockResponse(status_code=200, json_data={"matches_found": 0, "jobs": []})


def mock_request_with_jobs(*args, **kwargs):
    """Mock HTTP request returning jobs."""
    return MockResponse(
        status_code=200,
        json_data={
            "matches_found": 2,
            "jobs": [
                {"job_id": "job_123", "next_run_time": "2024-12-09T10:00:00"},
                {"job_id": "job_456", "next_run_time": "2024-12-09T11:00:00"},
            ],
        },
    )


@pytest.fixture
def fd_monitor():
    """Fixture to provide file descriptor monitoring."""
    monitor = FileDescriptorMonitor()
    monitor.start()
    yield monitor
    results = monitor.stop()
    # Optional: Add assertion here if you want all tests to enforce FD limits
    # assert results["leaked"] < 100, f"FD leak detected: {results['leaked']}"


@pytest.fixture
def mock_session():
    """Fixture to provide a mock session."""
    session = Mock()
    session.request.return_value = mock_request_success()
    session.adapters = {}
    return session


@pytest.mark.stress
class TestSessionStress:
    """Stress tests for session management."""

    @patch("bertrend.bertrend_apps.common.apscheduler_utils._get_session")
    def test_sequential_requests(self, mock_session_ctx, fd_monitor, mock_session):
        """Test sequential requests don't leak file descriptors."""
        mock_session_ctx.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_ctx.return_value.__exit__ = Mock(return_value=None)

        print("\n🔄 Running 1000 sequential requests...")

        for i in range(1000):
            _request("GET", "/jobs")
            if i % 100 == 0:
                current_fds = fd_monitor.update()
                print(f"  Request {i}: FDs = {current_fds}")

        time.sleep(0.5)  # Let cleanup happen
        results = fd_monitor.stop()

        assert results["leaked"] < 50, (
            f"Too many file descriptors leaked: {results['leaked']}"
        )

    @patch("bertrend.bertrend_apps.common.apscheduler_utils._get_session")
    def test_concurrent_requests(self, mock_session_ctx, fd_monitor, mock_session):
        """Test concurrent requests don't leak file descriptors."""
        mock_session_ctx.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_ctx.return_value.__exit__ = Mock(return_value=None)

        print("\n🔄 Running 500 concurrent requests (50 threads)...")

        def make_request(i):
            try:
                _request("GET", "/jobs")
                return i, "success"
            except Exception as e:
                return i, f"error: {e}"

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request, i) for i in range(500)]

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    current_fds = fd_monitor.update()
                    print(f"  Completed {completed}/500: FDs = {current_fds}")

        time.sleep(1)  # Let cleanup happen
        results = fd_monitor.stop()

        assert results["leaked"] < 100, (
            f"Too many file descriptors leaked: {results['leaked']}"
        )

    @patch("bertrend.bertrend_apps.common.apscheduler_utils._get_session")
    def test_rapid_fire_requests(self, mock_session_ctx, fd_monitor, mock_session):
        """Test rapid-fire requests without delays."""
        mock_session_ctx.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_ctx.return_value.__exit__ = Mock(return_value=None)

        print("\n🔄 Running 2000 rapid-fire requests...")

        errors = 0
        for i in range(2000):
            try:
                _request("GET", "/jobs")
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"  Error at request {i}: {e}")

            if i % 200 == 0:
                current_fds = fd_monitor.update()
                print(f"  Request {i}: FDs = {current_fds}, Errors = {errors}")

        time.sleep(1)
        results = fd_monitor.stop()

        print(f"Total errors: {errors}")
        assert results["leaked"] < 100, (
            f"Too many file descriptors leaked: {results['leaked']}"
        )

    @patch("bertrend.bertrend_apps.common.apscheduler_utils._request")
    def test_apscheduler_utils_methods(self, mock_request_func, fd_monitor):
        """Test APSchedulerUtils methods under load."""
        mock_request_func.return_value = mock_request_with_jobs()

        print("\n🔄 Testing APSchedulerUtils methods...")

        utils = APSchedulerUtils()

        # Test find_jobs repeatedly
        print("  Testing find_jobs (500 iterations)...")
        for i in range(500):
            utils.find_jobs(
                patterns={
                    "url": ".*/scrape-feed.*",
                    "json_data": {"user": "test", "model_id": "model1"},
                }
            )
            if i % 100 == 0:
                current_fds = fd_monitor.update()
                print(f"    Iteration {i}: FDs = {current_fds}")

        # Test find_jobs_description repeatedly
        print("  Testing find_jobs_description (500 iterations)...")
        for i in range(500):
            utils.find_jobs_description(
                patterns={
                    "url": ".*/train-new-model.*",
                    "json_data": {"user": "test", "model_id": "model2"},
                }
            )
            if i % 100 == 0:
                current_fds = fd_monitor.update()
                print(f"    Iteration {i}: FDs = {current_fds}")

        time.sleep(1)
        results = fd_monitor.stop()

        assert results["leaked"] < 100, (
            f"Too many file descriptors leaked: {results['leaked']}"
        )

    @patch("bertrend.bertrend_apps.common.apscheduler_utils._get_session")
    def test_error_handling_leak(self, mock_session_ctx, fd_monitor):
        """Test that errors don't cause FD leaks."""
        mock_session = Mock()
        call_count = [0]

        def side_effect_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise ConnectionError("Simulated connection error")
            return mock_request_success()

        mock_session.request.side_effect = side_effect_request
        mock_session_ctx.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_ctx.return_value.__exit__ = Mock(return_value=None)
        mock_session.adapters = {}

        print("\n🔄 Testing error handling (expecting ~333 errors out of 1000)...")

        errors = 0
        successes = 0
        for i in range(1000):
            try:
                _request("GET", "/jobs")
                successes += 1
            except Exception:
                errors += 1

            if i % 200 == 0:
                current_fds = fd_monitor.update()
                print(
                    f"  Request {i}: FDs = {current_fds}, "
                    f"Errors = {errors}, Successes = {successes}"
                )

        time.sleep(1)
        results = fd_monitor.stop()

        print(f"Final: {errors} errors, {successes} successes")
        assert results["leaked"] < 100, (
            f"Too many file descriptors leaked despite errors: {results['leaked']}"
        )
        assert errors > 0, "Expected some errors but got none"
        assert successes > 0, "Expected some successes but got none"
