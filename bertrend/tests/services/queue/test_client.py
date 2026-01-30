import pytest
import json
from unittest.mock import MagicMock, patch
from bertrend.services.queue.client import BertrendClient
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig


@pytest.fixture
def mock_config():
    return RabbitMQConfig()


@pytest.fixture
def client(mock_config):
    with patch("bertrend.services.queue.client.QueueManager") as mock_qm:
        client = BertrendClient(mock_config)
        client.queue_manager = mock_qm.return_value
        yield client


def test_send_request(client):
    client.queue_manager.publish_request.return_value = "test-corr-id"

    corr_id = client.send_request("/test", {"data": 1}, priority=7)

    assert corr_id == "test-corr-id"
    client.queue_manager.publish_request.assert_called_once_with(
        request_data={"endpoint": "/test", "method": "POST", "json_data": {"data": 1}},
        priority=7,
    )


def test_scrape_feed(client):
    client.queue_manager.publish_request.return_value = "corr-id"

    client.scrape_feed("feed.toml", user="test-user")

    client.queue_manager.publish_request.assert_called_once()
    args, kwargs = client.queue_manager.publish_request.call_args
    request_data = kwargs.get("request_data") or args[0]
    assert request_data["endpoint"] == "/scrape-feed"
    assert request_data["json_data"]["feed_cfg"] == "feed.toml"
    assert request_data["json_data"]["user"] == "test-user"


def test_train_new_model(client):
    client.queue_manager.publish_request.return_value = "corr-id"

    client.train_new_model(user="test-user", model_id="test-model")

    client.queue_manager.publish_request.assert_called_once()
    args, kwargs = client.queue_manager.publish_request.call_args
    request_data = kwargs.get("request_data") or args[0]
    assert request_data["endpoint"] == "/train-new-model"
    assert request_data["json_data"]["user"] == "test-user"
    assert request_data["json_data"]["model_id"] == "test-model"


def test_get_response(client):
    # Mocking the consumption of a response
    mock_ch = MagicMock()
    client.queue_manager.channel = mock_ch

    # Simulate a response being received
    def mock_basic_consume(queue, on_message_callback, auto_ack):
        # Create a mock message
        method = MagicMock()
        properties = MagicMock(correlation_id="test-corr-id")
        body = json.dumps({"status": "success", "result": "done"}).encode()
        on_message_callback(mock_ch, method, properties, body)

    mock_ch.basic_consume.side_effect = mock_basic_consume

    response = client.get_response("test-corr-id", timeout=1)

    assert response["status"] == "success"
    assert response["result"] == "done"
    mock_ch.stop_consuming.assert_called_once()
