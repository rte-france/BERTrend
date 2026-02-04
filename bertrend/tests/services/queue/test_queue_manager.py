from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest

from bertrend_apps.services.queue.queue_manager import QueueManager
from bertrend_apps.services.queue.rabbitmq_config import RabbitMQConfig


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
async def mock_aio_pika():
    with patch("aio_pika.connect_robust", new_callable=AsyncMock) as mock_connect:
        mock_conn = AsyncMock()
        mock_channel = AsyncMock()
        mock_connect.return_value = mock_conn
        mock_conn.channel.return_value = mock_channel
        yield mock_connect


@pytest.mark.asyncio
async def test_queue_manager_connect(mock_config, mock_aio_pika):
    qm = QueueManager(mock_config)
    await qm.connect()

    mock_aio_pika.assert_called_once()
    assert qm.connection is not None
    assert qm.channel is not None

    # Verify queue declarations
    qm.channel.declare_queue.assert_any_call(
        mock_config.request_queue,
        durable=True,
        arguments={
            "x-max-priority": 10,
            "x-message-ttl": 3600000,
            "x-dead-letter-exchange": "bertrend_dlx",
            "x-dead-letter-routing-key": "failed",
        },
    )
    qm.channel.declare_queue.assert_any_call(mock_config.response_queue, durable=True)


@pytest.mark.asyncio
async def test_publish_request(mock_config, mock_aio_pika):
    qm = QueueManager(mock_config)
    await qm.connect()

    request_data = {"task": "test"}
    correlation_id = await qm.publish_request(request_data, priority=8)

    assert correlation_id is not None
    qm.channel.default_exchange.publish.assert_called_once()

    args, kwargs = qm.channel.default_exchange.publish.call_args
    message = args[0]
    assert kwargs["routing_key"] == mock_config.request_queue
    assert message.body == msgpack.packb(request_data)
    assert message.priority == 8
    assert message.correlation_id == correlation_id
    assert message.reply_to == mock_config.response_queue


@pytest.mark.asyncio
async def test_publish_response(mock_config, mock_aio_pika):
    qm = QueueManager(mock_config)
    await qm.connect()

    response_data = {"result": "success"}
    correlation_id = "test-corr-id"
    await qm.publish_response(response_data, correlation_id)

    qm.channel.default_exchange.publish.assert_called_once()
    args, kwargs = qm.channel.default_exchange.publish.call_args
    message = args[0]
    assert kwargs["routing_key"] == mock_config.response_queue
    assert message.body == msgpack.packb(response_data)
    assert message.correlation_id == correlation_id


@pytest.mark.asyncio
async def test_consume_requests(mock_config, mock_aio_pika):
    qm = QueueManager(mock_config)
    await qm.connect()

    mock_queue = AsyncMock()
    qm.channel.get_queue.return_value = mock_queue

    callback = MagicMock()
    await qm.consume_requests(callback, prefetch_count=5)

    qm.channel.set_qos.assert_called_with(prefetch_count=5)
    qm.channel.get_queue.assert_called_with(mock_config.request_queue)
    mock_queue.consume.assert_called_once_with(callback, no_ack=False)


@pytest.mark.asyncio
async def test_close(mock_config, mock_aio_pika):
    qm = QueueManager(mock_config)
    await qm.connect()

    qm.connection.is_closed = False

    await qm.close()
    qm.connection.close.assert_called_once()
