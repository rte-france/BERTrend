#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Unit tests for QueueManager — focusing on republish_with_retry retry-limit enforcement.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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
    msg.body = json.dumps(body).encode("utf-8")
    msg.correlation_id = correlation_id
    msg.reply_to = reply_to
    msg.headers = headers or {}
    msg.priority = 5
    msg.content_type = "application/json"
    msg.ack = AsyncMock()
    msg.nack = AsyncMock()
    msg.reject = AsyncMock()
    return msg


def _make_queue_manager(max_retries: int = 2) -> QueueManager:
    config = RabbitMQConfig()
    config.max_retries = max_retries
    qm = QueueManager(config)
    qm.channel = MagicMock()
    qm.channel.is_closed = False
    qm.channel.default_exchange = MagicMock()
    qm.channel.default_exchange.publish = AsyncMock()
    return qm


# ---------------------------------------------------------------------------
# Test 1 — republish_with_retry returns False and does not publish at limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_republish_with_retry_stops_at_max_retries():
    """
    republish_with_retry must return False and not publish when
    retry_count + 1 would exceed max_retries.
    """
    qm = _make_queue_manager(max_retries=2)
    msg = _make_message({}, headers={"x-retry-count": 2})

    result = await qm.republish_with_retry(msg, retry_count=2)

    assert result is False
    qm.channel.default_exchange.publish.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 2 — republish_with_retry returns True and publishes below limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_republish_with_retry_publishes_below_max_retries():
    """
    republish_with_retry must return True and publish when
    retry_count + 1 is still within max_retries.
    """
    qm = _make_queue_manager(max_retries=2)
    msg = _make_message({}, headers={"x-retry-count": 1})

    result = await qm.republish_with_retry(msg, retry_count=1)

    assert result is True
    qm.channel.default_exchange.publish.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 3 — republish_with_retry increments x-retry-count in the new message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_republish_with_retry_increments_header():
    """
    The republished message must carry x-retry-count = retry_count + 1.
    """
    qm = _make_queue_manager(max_retries=3)
    msg = _make_message({}, headers={"x-retry-count": 0})

    await qm.republish_with_retry(msg, retry_count=0)

    call_args = qm.channel.default_exchange.publish.call_args
    published_message = call_args[0][0]  # first positional arg
    assert published_message.headers["x-retry-count"] == 1


# ---------------------------------------------------------------------------
# Test 4 — republish_with_retry reconnects if channel is closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_republish_with_retry_reconnects_if_channel_closed():
    """
    If the channel is closed, republish_with_retry must call connect() before
    publishing.
    """
    qm = _make_queue_manager(max_retries=2)
    qm.channel.is_closed = True
    qm.connect = AsyncMock()

    # After connect(), set up a fresh channel mock
    async def fake_connect():
        qm.channel = MagicMock()
        qm.channel.is_closed = False
        qm.channel.default_exchange = MagicMock()
        qm.channel.default_exchange.publish = AsyncMock()

    qm.connect.side_effect = fake_connect

    msg = _make_message({}, headers={"x-retry-count": 0})
    result = await qm.republish_with_retry(msg, retry_count=0)

    qm.connect.assert_awaited_once()
    assert result is True
    qm.channel.default_exchange.publish.assert_awaited_once()
