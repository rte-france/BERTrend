#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Unit tests for BertrendWorker.callback() — focusing on:
  1. Job timeout: a hung process_request is abandoned and the message is
     dead-lettered without retry (its thread cannot be stopped).
  2. Queue blocking: with prefetch_count=1 an unacked message blocks subsequent
     messages from being delivered (RabbitMQ behaviour, simulated here).
  3. Retry limit: messages are rejected to DLQ after max_retries attempts.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bertrend.bertrend_apps.services.queue_management.bertrend_worker import (
    BertrendWorker,
)
from bertrend.bertrend_apps.services.queue_management.queue_manager import QueueManager
from bertrend.bertrend_apps.services.queue_management.rabbitmq_config import (
    RabbitMQConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    body: dict,
    correlation_id: str = "test-corr-id",
    reply_to: str | None = None,
    headers: dict | None = None,
):
    """Build a minimal mock of aio_pika.abc.AbstractIncomingMessage."""
    msg = MagicMock()
    msg.body = (
        json.dumps(body).decode
        if isinstance(body, str)
        else json.dumps(body).encode("utf-8")
    )
    msg.correlation_id = correlation_id
    msg.reply_to = reply_to
    msg.headers = headers or {}
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
# Test 1 — Timeout: message is dead-lettered without retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_does_not_retry_on_timeout():
    """
    When process_request takes longer than job_timeout, the job thread keeps
    running in the background, so the message must not be republished (that
    would start a concurrent duplicate run): it goes to the DLQ once.
    """
    worker = _make_worker(job_timeout=1)  # 1-second timeout for fast test
    worker.queue_manager.republish_with_retry = AsyncMock(return_value=True)

    async def slow_handler(request_data):
        await asyncio.sleep(10)  # much longer than timeout
        return {"status": "success", "response": {}}

    msg = _make_message(
        {"endpoint": "/train-new-model", "method": "POST", "json_data": {}}
    )

    with patch.object(worker, "process_request", side_effect=slow_handler):
        await worker.callback(msg)

    worker.queue_manager.republish_with_retry.assert_not_awaited()
    msg.reject.assert_awaited_once_with(requeue=False)
    msg.ack.assert_not_awaited()
    msg.nack.assert_not_awaited()


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
    msg2 = _make_message(
        {"endpoint": "/scrape", "method": "POST", "json_data": {}},
        correlation_id="corr-2",
    )

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


# ---------------------------------------------------------------------------
# Test 5 — Retry limit: nack on first failures, reject to DLQ on last attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nack_or_discard_requeues_below_max_retries():
    """
    When retry_count < max_retries, a failing message must be republished with
    an incremented x-retry-count header and the original acked, so that only
    the final failed attempt reaches the DLQ.
    """
    worker = _make_worker(job_timeout=5)
    worker.config.max_retries = 2
    worker.queue_manager.republish_with_retry = AsyncMock(return_value=True)

    async def failing_handler(request_data):
        raise RuntimeError("transient error")

    # First attempt: x-retry-count header absent (defaults to 0)
    msg = _make_message(
        {"endpoint": "/scrape", "method": "POST", "json_data": {}},
        headers={},
    )

    with patch.object(worker, "process_request", side_effect=failing_handler):
        await worker.callback(msg)

    worker.queue_manager.republish_with_retry.assert_awaited_once_with(msg, 0)
    msg.ack.assert_awaited_once()
    msg.reject.assert_not_awaited()
    msg.nack.assert_not_awaited()


@pytest.mark.asyncio
async def test_failing_job_reaches_dlq_only_once():
    """
    A job failing on every attempt must leave a single copy in the DLQ, not
    one per attempt. Drives the real QueueManager.republish_with_retry and
    feeds each republished copy back to the worker, as RabbitMQ would.
    """
    worker = _make_worker(job_timeout=5)
    worker.config.max_retries = 2
    queue_manager = QueueManager(worker.config)
    queue_manager.channel = MagicMock()
    queue_manager.channel.is_closed = False
    queue_manager.channel.default_exchange.publish = AsyncMock()
    worker.queue_manager = queue_manager

    async def failing_handler(request_data):
        raise RuntimeError("persistent error")

    body = {"endpoint": "/scrape", "method": "POST", "json_data": {}}
    msg = _make_message(body, headers={})
    delivered = [msg]

    with patch.object(worker, "process_request", side_effect=failing_handler):
        while True:
            await worker.callback(msg)
            publish = queue_manager.channel.default_exchange.publish
            if publish.await_count < len(delivered):
                break
            republished = publish.await_args.args[0]
            msg = _make_message(body, headers=republished.headers)
            delivered.append(msg)

    assert len(delivered) == worker.config.max_retries + 1
    dead_lettered = [m for m in delivered if m.reject.await_count]
    assert dead_lettered == [delivered[-1]]
    delivered[-1].reject.assert_awaited_once_with(requeue=False)
    for m in delivered[:-1]:
        m.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_nack_or_discard_rejects_at_max_retries():
    """
    When retry_count >= max_retries, republish_with_retry returns False.
    _nack_or_discard must still call reject(requeue=False) to send to DLQ.
    """
    worker = _make_worker(job_timeout=5)
    worker.config.max_retries = 2
    # Simulate limit already reached
    worker.queue_manager.republish_with_retry = AsyncMock(return_value=False)

    async def failing_handler(request_data):
        raise RuntimeError("persistent error")

    # Simulate message that has already been retried max_retries times
    msg = _make_message(
        {"endpoint": "/scrape", "method": "POST", "json_data": {}},
        headers={"x-retry-count": 2},
    )

    with patch.object(worker, "process_request", side_effect=failing_handler):
        await worker.callback(msg)

    worker.queue_manager.republish_with_retry.assert_awaited_once_with(msg, 2)
    msg.reject.assert_awaited_once_with(requeue=False)
    msg.nack.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_rejects_at_max_retries():
    """
    A timed-out message that was already retried (after an earlier exception)
    must be rejected to the DLQ without another retry.
    """
    worker = _make_worker(job_timeout=1)
    worker.config.max_retries = 2
    # Simulate limit already reached: republish_with_retry returns False
    worker.queue_manager.republish_with_retry = AsyncMock(return_value=False)

    async def slow_handler(request_data):
        await asyncio.sleep(10)
        return {"status": "success", "response": {}}

    msg = _make_message(
        {"endpoint": "/train-new-model", "method": "POST", "json_data": {}},
        headers={"x-retry-count": 2},
    )

    with patch.object(worker, "process_request", side_effect=slow_handler):
        await worker.callback(msg)

    worker.queue_manager.republish_with_retry.assert_not_awaited()
    msg.reject.assert_awaited_once_with(requeue=False)
    msg.nack.assert_not_awaited()
