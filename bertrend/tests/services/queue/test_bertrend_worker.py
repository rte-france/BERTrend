import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from bertrend.services.queue.bertrend_worker import BertrendWorker, ENDPOINT_HANDLERS
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig


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
        "bertrend.services.queue.bertrend_worker.ENDPOINT_HANDLERS",
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
        "bertrend.services.queue.bertrend_worker.ENDPOINT_HANDLERS",
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
    mock_message.body = json.dumps({"endpoint": "/test", "json_data": {}}).encode()

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
async def test_start(worker):
    worker.queue_manager.connect = AsyncMock()
    worker.queue_manager.consume_requests = AsyncMock()
    worker.queue_manager.close = AsyncMock()

    # start() now waits on a future. We can use wait_for to timeout.
    with patch("asyncio.Future", return_value=asyncio.get_event_loop().create_future()) as mock_future:
        # Resolve the future immediately or after a short time
        f = mock_future.return_value
        f.set_result(None)
        
        await worker.start()

    worker.queue_manager.connect.assert_called_once()
    worker.queue_manager.consume_requests.assert_called_once()
    worker.queue_manager.close.assert_called_once()
