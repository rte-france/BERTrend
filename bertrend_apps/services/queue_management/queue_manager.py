# bertrend_apps/services/queue_manager.py

import json
from typing import Callable

import aio_pika
from loguru import logger

from bertrend_apps.services.queue_management.rabbitmq_config import RabbitMQConfig


class QueueManager:
    """Manages RabbitMQ connections and operations"""

    def __init__(self, config: RabbitMQConfig):
        self.config = config
        self.connection: aio_pika.abc.AbstractConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self):
        """Establish connection to RabbitMQ"""
        url = (
            f"amqp://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.virtual_host.lstrip('/')}"
        )

        self.connection = await aio_pika.connect_robust(
            url,
            heartbeat=600,
            timeout=300,
        )
        self.channel = await self.connection.channel()

        # Declare queues with retry mechanism
        await self._setup_queues()

        logger.info(f"Connected to RabbitMQ at {self.config.host}:{self.config.port}")

    async def _setup_queues(self):
        """Setup main and retry queues"""
        # Main request queue
        await self.channel.declare_queue(
            self.config.request_queue,
            durable=True,  # Survive broker restart
            arguments={
                "x-max-priority": 10,  # Enable priority queue
                "x-message-ttl": 3600000,  # 1 hour TTL
                # Dead letter exchange for failed messages
                "x-dead-letter-exchange": "bertrend_dlx",
                "x-dead-letter-routing-key": "failed",
            },
        )

        # Response queue
        await self.channel.declare_queue(self.config.response_queue, durable=True)

        # Dead letter queue for failed requests
        await self.channel.declare_exchange(
            name="bertrend_dlx", type=aio_pika.ExchangeType.DIRECT, durable=True
        )

        queue_failed = await self.channel.declare_queue("bertrend_failed", durable=True)

        await queue_failed.bind(exchange="bertrend_dlx", routing_key="failed")

        logger.info("Queues configured successfully")

    async def publish_request(
        self,
        request_data: dict,
        priority: int = 5,
        correlation_id: str | None = None,
    ) -> str:
        """Publish a request to the queue_management"""
        if not self.channel or self.channel.is_closed:
            await self.connect()

        # Generate correlation ID if not provided
        if not correlation_id:
            import uuid

            correlation_id = str(uuid.uuid4())

        # Prepare message
        message_body = json.dumps(request_data).encode("utf-8")

        # Publish with properties
        message = aio_pika.Message(
            body=message_body,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            priority=priority,
            correlation_id=correlation_id,
            content_type="application/json",
            reply_to=self.config.response_queue,
        )

        await self.channel.default_exchange.publish(
            message,
            routing_key=self.config.request_queue,
        )

        logger.info(f"Published request with correlation_id: {correlation_id}")
        return correlation_id

    async def consume_requests(
        self, callback: Callable, prefetch_count: int | None = None
    ):
        """Start consuming requests from the queue_management"""
        if not self.channel or self.channel.is_closed:
            await self.connect()

        # Set QoS - only process one message at a time per worker
        prefetch = prefetch_count or self.config.prefetch_count
        await self.channel.set_qos(prefetch_count=prefetch)

        # Setup consumer
        queue = await self.channel.get_queue(self.config.request_queue)

        logger.info(f"Started consuming from {self.config.request_queue}")
        logger.info(f"Prefetch count: {prefetch}")

        await queue.consume(callback, no_ack=False)

    async def publish_response(self, response_data: dict, correlation_id: str):
        """Publish response to response queue_management"""
        if not self.channel or self.channel.is_closed:
            await self.connect()

        message_body = json.dumps(response_data).encode("utf-8")

        message = aio_pika.Message(
            body=message_body,
            correlation_id=correlation_id,
            content_type="application/json",
        )

        await self.channel.default_exchange.publish(
            message,
            routing_key=self.config.response_queue,
        )

        logger.info(f"Published response for correlation_id: {correlation_id}")

    async def close(self):
        """Close connection"""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("RabbitMQ connection closed")
