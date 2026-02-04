from unittest.mock import AsyncMock, MagicMock, patch

import json
import pytest

from bertrend_apps.services.queue.bertrend_worker import BertrendWorker
from bertrend_apps.services.queue.rabbitmq_config import RabbitMQConfig


@pytest.fixture
def mock_config():
    return RabbitMQConfig()


@pytest.fixture
def worker(mock_config):
    return BertrendWorker(mock_config)


@pytest.mark.asyncio
async def test_process_request_success(worker):
    # Mock a handler function
    mock_handler = AsyncMock(return_value={"status": "success", "data": "test_result"})
    mock_model = MagicMock()

    with patch.dict(
        "bertrend_apps.services.queue.bertrend_worker.ENDPOINT_HANDLERS",
        {"/test": (mock_handler, mock_model)},
    ):
        request_data = {
            "endpoint": "/test",
            "method": "POST",
            "json_data": {"key": "value"},
        }

        response = await worker.process_request(request_data)

        assert response["status"] == "success"
        assert response["response"] == {"status": "success", "data": "test_result"}
        mock_handler.assert_called_once()
        mock_model.assert_called_once_with(key="value")


@pytest.mark.asyncio
async def test_process_request_unknown_endpoint(worker):
    request_data = {"endpoint": "/unknown", "method": "POST", "json_data": {}}

    response = await worker.process_request(request_data)

    assert response["status"] == "error"
    assert "Unknown endpoint" in response["error"]


@pytest.mark.asyncio
async def test_process_request_invalid_data(worker):
    mock_handler = AsyncMock()
    # Mock model that raises exception on instantiation
    mock_model = MagicMock(side_effect=ValueError("Invalid data"))

    with patch.dict(
        "bertrend_apps.services.queue.bertrend_worker.ENDPOINT_HANDLERS",
        {"/test": (mock_handler, mock_model)},
    ):
        request_data = {"endpoint": "/test", "json_data": {"bad": "data"}}

        response = await worker.process_request(request_data)

        assert response["status"] == "error"
        assert "Invalid request data" in response["error"]


@pytest.mark.asyncio
async def test_callback(worker):
    # Now callback is async
    mock_message = AsyncMock()
    mock_message.correlation_id = "test-corr-id"
    mock_message.reply_to = "response_queue"
    mock_message.body = json.dumps({"endpoint": "/test", "json_data": {}}).encode(
        "utf-8"
    )

    # Mock process_request
    worker.process_request = AsyncMock(return_value={"status": "ok"})
    worker.queue_manager.publish_response = AsyncMock()

    # Run the callback
    await worker.callback(mock_message)

    # Verify processing
    worker.process_request.assert_called_once()
    worker.queue_manager.publish_response.assert_called_once()

    # Verify acknowledgment
    mock_message.ack.assert_called_once()


@pytest.mark.asyncio
async def test_callback_json_error(worker):
    mock_message = AsyncMock()
    mock_message.correlation_id = "test-corr-id"
    mock_message.body = b"invalid json"

    await worker.callback(mock_message)

    mock_message.reject.assert_called_once_with(requeue=False)


@pytest.mark.asyncio
async def test_handle_scrape_direct_call():
    from bertrend_apps.services.models.data_provider_models import ScrapeRequest
    from bertrend_apps.services.queue.bertrend_worker import handle_scrape

    req = ScrapeRequest(
        keywords="ai",
        provider="google",
        after="2025-01-01",
        before="2025-01-02",
        max_results=10,
    )

    with patch(
        "bertrend_apps.services.queue.bertrend_worker.asyncio.to_thread",
        new_callable=AsyncMock,
    ) as mock_to_thread:
        mock_to_thread.return_value = [{"title": "test"}]
        result = await handle_scrape(req)

        assert result == [{"title": "test"}]
        mock_to_thread.assert_called_once()
