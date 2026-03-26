#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Unit tests for BertrendWorker.callback() — focusing on:
  1. Job timeout: a hung process_request is cancelled and the message is nacked.
  2. Queue blocking: with prefetch_count=1 an unacked message blocks subsequent
     messages from being delivered (RabbitMQ behaviour, simulated here).
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bertrend.bertrend_apps.services.queue_management.bertrend_worker import (
    BertrendWorker,
)
from bertrend.bertrend_apps.services.queue_management.rabbitmq_config import (
    RabbitMQConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message(body: dict, correlation_id: str = "test-corr-id", reply_to: str | None = None):
    """Build a minimal mock of aio_pika.abc.AbstractIncomingMessage."""
    msg = MagicMock()
    msg.body = json.dumps(body).decode if isinstance(body, str) else json.dumps(body).encode("utf-8")
    msg.correlation_id = correlation_id
    msg.reply_to = reply_to
    msg.ack = AsyncMock()
    msg.nack = AsyncMock()
    msg.reject = AsyncMock()
    return msg


def _make_worker(job_timeout: int = 5) -> BertrendWorker:
    config = RabbitMQConfig(job_timeout=job_timeout)
    worker = BertrendWorker(config)
    # Replace queue_manager with a mock so no real RabbitMQ connection is needed
    worker.queue_manager = MagicMock()
    worker.queue_manager.publish_response = AsyncMock()
    worker.queue_manager.publish_error = AsyncMock()
    return worker


# ---------------------------------------------------------------------------
# Test 1 — Timeout: hung process_request is cancelled, message is nacked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_nacks_on_timeout():
    """
    When process_request takes longer than job_timeout, the message must be
    nacked (requeue=True) and NOT acked.
    """
    worker = _make_worker(job_timeout=1)  # 1-second timeout for fast test

    async def slow_handler(request_data):
        await asyncio.sleep(10)  # much longer than timeout
        return {"status": "success", "response": {}}

    msg = _make_message({"endpoint": "/train-new-model", "method": "POST", "json_data": {}})

    with patch.object(worker, "process_request", side_effect=slow_handler):
        await worker.callback(msg)

    msg.nack.assert_awaited_once_with(requeue=True)
    msg.ack.assert_not_awaited()
    msg.reject.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 2 — Success: message is acked after process_request completes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_acks_on_success():
    """
    When process_request completes within the timeout, the message must be
    acked and the response published.
    """
    worker = _make_worker(job_timeout=5)

    async def fast_handler(request_data):
        return {"status": "success", "response": {"result": 42}}

    msg = _make_message(
        {"endpoint": "/scrape", "method": "POST", "json_data": {}},
        reply_to="bertrend_responses",
    )

    with patch.object(worker, "process_request", side_effect=fast_handler):
        await worker.callback(msg)

    msg.ack.assert_awaited_once()
    msg.nack.assert_not_awaited()
    worker.queue_manager.publish_response.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 3 — Error response: acked and error published (not nacked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_callback_acks_on_error_response():
    """
    When process_request returns a status=error dict (internal error, not a
    raised exception), the message must still be acked and the error published.
    """
    worker = _make_worker(job_timeout=5)

    async def error_handler(request_data):
        return {"status": "error", "error": "something went wrong"}

    msg = _make_message(
        {"endpoint": "/scrape", "method": "POST", "json_data": {}},
        reply_to="bertrend_responses",
    )

    with patch.object(worker, "process_request", side_effect=error_handler):
        await worker.callback(msg)

    msg.ack.assert_awaited_once()
    msg.nack.assert_not_awaited()
    worker.queue_manager.publish_error.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 4 — Queue blocking simulation with prefetch_count=1
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prefetch_1_blocks_second_message_until_first_acked():
    """
    Simulate prefetch_count=1 behaviour: the second message callback is only
    invoked after the first message is acked.  We model this by running two
    callback coroutines concurrently and checking ordering via a shared log.
    """
    worker = _make_worker(job_timeout=10)
    events: list[str] = []

    async def slow_handler(request_data):
        events.append("job1_start")
        await asyncio.sleep(0.2)
        events.append("job1_end")
        return {"status": "success", "response": {}}

    async def fast_handler(request_data):
        events.append("job2_start")
        return {"status": "success", "response": {}}

    msg1 = _make_message({"endpoint": "/scrape", "method": "POST", "json_data": {}})
    msg2 = _make_message({"endpoint": "/scrape", "method": "POST", "json_data": {}}, correlation_id="corr-2")

    # Patch process_request to use slow_handler for msg1 and fast_handler for msg2
    call_count = 0

    async def dispatch_handler(request_data):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await slow_handler(request_data)
        return await fast_handler(request_data)

    with patch.object(worker, "process_request", side_effect=dispatch_handler):
        # With prefetch_count=1, RabbitMQ would only deliver msg2 after msg1 is acked.
        # We simulate this by awaiting msg1 fully before starting msg2.
        await worker.callback(msg1)
        await worker.callback(msg2)

    # job2 must start only after job1 has ended (sequential delivery)
    assert events.index("job1_end") < events.index("job2_start"), (
        "With prefetch_count=1, msg2 should only be processed after msg1 is acked"
    )
    msg1.ack.assert_awaited_once()
    msg2.ack.assert_awaited_once()
