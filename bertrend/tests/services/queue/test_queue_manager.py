import pytest
from unittest.mock import MagicMock, patch
import json
import pika
from bertrend.services.queue.queue_manager import QueueManager
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig


@pytest.fixture
def mock_config():
    return RabbitMQConfig(
        host="localhost",
        port=5672,
        username="guest",
        password="guest",
        virtual_host="/",
    )


@pytest.fixture
def mock_pika():
    with patch("pika.BlockingConnection") as mock_conn:
        yield mock_conn


def test_queue_manager_connect(mock_config, mock_pika):
    qm = QueueManager(mock_config)
    qm.connect()

    mock_pika.assert_called_once()
    assert qm.connection is not None
    assert qm.channel is not None

    # Verify queue declarations
    qm.channel.queue_declare.assert_any_call(
        queue=mock_config.request_queue,
        durable=True,
        arguments={
            "x-max-priority": 10,
            "x-message-ttl": 3600000,
            "x-dead-letter-exchange": "bertrend_dlx",
            "x-dead-letter-routing-key": "failed",
        },
    )
    qm.channel.queue_declare.assert_any_call(
        queue=mock_config.response_queue, durable=True
    )


def test_publish_request(mock_config, mock_pika):
    qm = QueueManager(mock_config)
    # Mock connect to avoid real pika calls
    qm.connect = MagicMock()
    qm.channel = MagicMock()

    request_data = {"task": "test"}
    correlation_id = qm.publish_request(request_data, priority=8)

    assert correlation_id is not None
    qm.channel.basic_publish.assert_called_once()

    args, kwargs = qm.channel.basic_publish.call_args
    assert kwargs["routing_key"] == mock_config.request_queue
    assert kwargs["body"] == json.dumps(request_data)

    properties = kwargs["properties"]
    assert properties.priority == 8
    assert properties.correlation_id == correlation_id
    assert properties.reply_to == mock_config.response_queue


def test_publish_response(mock_config, mock_pika):
    qm = QueueManager(mock_config)
    qm.connect = MagicMock()
    qm.channel = MagicMock()

    response_data = {"result": "success"}
    correlation_id = "test-corr-id"
    qm.publish_response(response_data, correlation_id)

    qm.channel.basic_publish.assert_called_once()
    args, kwargs = qm.channel.basic_publish.call_args
    assert kwargs["routing_key"] == mock_config.response_queue
    assert kwargs["body"] == json.dumps(response_data)
    assert kwargs["properties"].correlation_id == correlation_id


def test_consume_requests(mock_config, mock_pika):
    qm = QueueManager(mock_config)
    qm.connect = MagicMock()
    qm.channel = MagicMock()

    callback = MagicMock()
    qm.consume_requests(callback, prefetch_count=5)

    qm.channel.basic_qos.assert_called_with(prefetch_count=5)
    qm.channel.basic_consume.assert_called_once()
    assert qm.channel.basic_consume.call_args[1]["queue"] == mock_config.request_queue
    assert qm.channel.basic_consume.call_args[1]["on_message_callback"] == callback
    qm.channel.start_consuming.assert_called_once()


def test_close(mock_config, mock_pika):
    qm = QueueManager(mock_config)
    mock_conn = MagicMock()
    qm.connection = mock_conn
    mock_conn.is_closed = False

    qm.close()
    mock_conn.close.assert_called_once()
