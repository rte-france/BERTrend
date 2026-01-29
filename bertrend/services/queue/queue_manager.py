# bertrend_apps/services/queue_manager.py

import pika
import json
from typing import Optional, Callable

from loguru import logger

from bertrend.services.queue.rabbitmq_config import RabbitMQConfig


class QueueManager:
    """Manages RabbitMQ connections and operations"""

    def __init__(self, config: RabbitMQConfig):
        self.config = config
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None

    def connect(self):
        """Establish connection to RabbitMQ"""
        credentials = pika.PlainCredentials(self.config.username, self.config.password)

        parameters = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            virtual_host=self.config.virtual_host,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Declare queues with retry mechanism
        self._setup_queues()

        logger.info(f"Connected to RabbitMQ at {self.config.host}:{self.config.port}")

    def _setup_queues(self):
        """Setup main and retry queues"""
        # Main request queue
        self.channel.queue_declare(
            queue=self.config.request_queue,
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
        self.channel.queue_declare(queue=self.config.response_queue, durable=True)

        # Dead letter queue for failed requests
        self.channel.exchange_declare(
            exchange="bertrend_dlx", exchange_type="direct", durable=True
        )

        self.channel.queue_declare(queue="bertrend_failed", durable=True)

        self.channel.queue_bind(
            exchange="bertrend_dlx", queue="bertrend_failed", routing_key="failed"
        )

        logger.info("Queues configured successfully")

    def publish_request(
        self,
        request_data: dict,
        priority: int = 5,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Publish a request to the queue"""
        if not self.channel:
            self.connect()

        # Generate correlation ID if not provided
        if not correlation_id:
            import uuid

            correlation_id = str(uuid.uuid4())

        # Prepare message
        message = json.dumps(request_data)

        # Publish with properties
        properties = pika.BasicProperties(
            delivery_mode=2,  # Persistent message
            priority=priority,
            correlation_id=correlation_id,
            content_type="application/json",
            reply_to=self.config.response_queue,
        )

        self.channel.basic_publish(
            exchange="",
            routing_key=self.config.request_queue,
            body=message,
            properties=properties,
        )

        logger.info(f"Published request with correlation_id: {correlation_id}")
        return correlation_id

    def consume_requests(
        self, callback: Callable, prefetch_count: Optional[int] = None
    ):
        """Start consuming requests from the queue"""
        if not self.channel:
            self.connect()

        # Set QoS - only process one message at a time per worker
        prefetch = prefetch_count or self.config.prefetch_count
        self.channel.basic_qos(prefetch_count=prefetch)

        # Setup consumer
        self.channel.basic_consume(
            queue=self.config.request_queue,
            on_message_callback=callback,
            auto_ack=False,  # Manual acknowledgment for reliability
        )

        logger.info(f"Started consuming from {self.config.request_queue}")
        logger.info(f"Prefetch count: {prefetch}")

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.channel.stop_consuming()

    def publish_response(self, response_data: dict, correlation_id: str):
        """Publish response to response queue"""
        if not self.channel:
            self.connect()

        message = json.dumps(response_data)

        properties = pika.BasicProperties(
            correlation_id=correlation_id, content_type="application/json"
        )

        self.channel.basic_publish(
            exchange="",
            routing_key=self.config.response_queue,
            body=message,
            properties=properties,
        )

        logger.info(f"Published response for correlation_id: {correlation_id}")

    def close(self):
        """Close connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ connection closed")
