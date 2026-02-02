import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bertrend.services.queue.client import BertrendClient
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig


@pytest.fixture
def mock_config():
    return RabbitMQConfig()


@pytest.fixture
async def client(mock_config):
    with patch("bertrend.services.queue.client.QueueManager") as mock_qm_class:
        mock_qm = AsyncMock()
        mock_qm_class.return_value = mock_qm
        
        # Mock channel for _consume_responses
        mock_channel = AsyncMock()
        mock_qm.channel = mock_channel
        
        c = BertrendClient(mock_config)
        # Avoid starting the actual consumer task in tests unless needed
        with patch.object(BertrendClient, "_consume_responses", return_value=None):
            await c.connect()
        
        yield c
        await c.close()


@pytest.mark.asyncio
async def test_send_request(client):
    client.queue_manager.publish_request.return_value = "test-corr-id"

    corr_id = await client.send_request("/test", {"data": 1}, priority=7)

    assert corr_id == "test-corr-id"
    client.queue_manager.publish_request.assert_called_once_with(
        request_data={"endpoint": "/test", "method": "POST", "json_data": {"data": 1}},
        priority=7,
    )
    assert corr_id in client.pending_requests


@pytest.mark.asyncio
async def test_scrape_feed(client):
    client.queue_manager.publish_request.return_value = "corr-id"

    await client.scrape_feed("feed.toml", user="test-user")

    client.queue_manager.publish_request.assert_called_once()
    args, kwargs = client.queue_manager.publish_request.call_args
    request_data = kwargs.get("request_data") or args[0]
    assert request_data["endpoint"] == "/scrape-feed"
    assert request_data["json_data"]["feed_cfg"] == "feed.toml"
    assert request_data["json_data"]["user"] == "test-user"


@pytest.mark.asyncio
async def test_train_new_model(client):
    client.queue_manager.publish_request.return_value = "corr-id"

    await client.train_new_model(user="test-user", model_id="test-model")

    client.queue_manager.publish_request.assert_called_once()
    args, kwargs = client.queue_manager.publish_request.call_args
    request_data = kwargs.get("request_data") or args[0]
    assert request_data["endpoint"] == "/train-new-model"
    assert request_data["json_data"]["user"] == "test-user"
    assert request_data["json_data"]["model_id"] == "test-model"


@pytest.mark.asyncio
async def test_wait_for_response(client):
    correlation_id = "test-corr-id"
    future = asyncio.get_event_loop().create_future()
    client.pending_requests[correlation_id] = future
    
    # Simulate response being set in the future by the consumer task
    expected_response = {"status": "success", "result": "done"}
    
    async def set_response():
        await asyncio.sleep(0.1)
        future.set_result(expected_response)
    
    asyncio.create_task(set_response())
    
    response = await client.wait_for_response(correlation_id, timeout=1)

    assert response == expected_response
    assert correlation_id not in client.pending_requests
