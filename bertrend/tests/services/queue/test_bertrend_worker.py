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


def test_callback(worker):
    # This is a bit tricky as callback is sync but calls async process_request
    mock_ch = MagicMock()
    mock_method = MagicMock(delivery_tag=123)
    mock_props = MagicMock(correlation_id="test-corr-id", reply_to="response_queue")
    body = json.dumps({"endpoint": "/test", "json_data": {}}).encode()

    # Mock process_request to be used inside the callback's async runner
    worker.process_request = AsyncMock(return_value={"status": "ok"})
    worker.queue_manager.publish_response = MagicMock()

    # Run the callback
    worker.callback(mock_ch, mock_method, mock_props, body)

    # Verify processing
    worker.process_request.assert_called_once()
    worker.queue_manager.publish_response.assert_called_once()

    # Verify acknowledgment
    mock_ch.basic_ack.assert_called_once_with(delivery_tag=123)


def test_callback_json_error(worker):
    mock_ch = MagicMock()
    mock_method = MagicMock(delivery_tag=123)
    mock_props = MagicMock(correlation_id="test-corr-id")
    body = b"invalid json"

    worker.callback(mock_ch, mock_method, mock_props, body)

    mock_ch.basic_reject.assert_called_once_with(delivery_tag=123, requeue=False)


def test_start(worker):
    worker.queue_manager.connect = MagicMock()
    worker.queue_manager.consume_requests = MagicMock()
    worker.queue_manager.close = MagicMock()

    # start() calls consume_requests which is usually blocking.
    # We can mock it to return immediately.

    worker.start()

    worker.queue_manager.connect.assert_called_once()
    worker.queue_manager.consume_requests.assert_called_once()
    worker.queue_manager.close.assert_called_once()
