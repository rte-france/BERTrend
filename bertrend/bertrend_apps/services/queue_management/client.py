#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
BERTrend Client - Sends requests to BERTrend service via RabbitMQ queue_management.

This client simulates requests similar to what the scheduler service sends,
allowing testing of the queue_management-based execution flow.
"""

import asyncio
import json
from typing import Any

from loguru import logger

from bertrend.bertrend_apps.services.queue_management.queue_manager import QueueManager
from bertrend.bertrend_apps.services.queue_management.rabbitmq_config import (
    RabbitMQConfig,
)


class BertrendClient:
    """Client for sending requests to BERTrend service via RabbitMQ"""

    def __init__(self, config: RabbitMQConfig | None = None):
        self.config = config or RabbitMQConfig()
        self.queue_manager = QueueManager(self.config)
        self.pending_requests: dict[str, asyncio.Future] = {}
        self._response_consumer_task: asyncio.Task | None = None

    async def connect(self):
        """Establish connection and start response consumer"""
        await self.queue_manager.connect()
        self._response_consumer_task = asyncio.create_task(self._consume_responses())

    async def _consume_responses(self):
        """Background task to consume responses from the response queue_management"""
        if not self.queue_manager.channel:
            await self.queue_manager.connect()

        queue = await self.queue_manager.channel.get_queue(self.config.response_queue)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    correlation_id = message.correlation_id
                    if correlation_id in self.pending_requests:
                        future = self.pending_requests[correlation_id]
                        if not future.done():
                            try:
                                response = json.loads(message.body.decode("utf-8"))
                                future.set_result(response)
                            except Exception as e:
                                future.set_exception(e)
                        del self.pending_requests[correlation_id]

    async def send_request(
        self,
        endpoint: str,
        json_data: dict[str, Any],
        method: str = "POST",
        priority: int = 5,
    ) -> str:
        """
        Send a request to the BERTrend service via the queue_management.

        Args:
            endpoint: The API endpoint path (e.g., "/scrape-feed", "/train-new-model")
            json_data: The request body data
            method: HTTP method (default: POST)
            priority: Request priority (0-10, higher = more priority)

        Returns:
            correlation_id for tracking the request
        """
        request_data = {
            "endpoint": endpoint,
            "method": method,
            "json_data": json_data,
        }

        correlation_id = await self.queue_manager.publish_request(
            request_data=request_data, priority=priority
        )

        self.pending_requests[correlation_id] = asyncio.get_event_loop().create_future()
        logger.info(f"Sent request to {endpoint} with correlation_id: {correlation_id}")

        return correlation_id

    async def scrape_feed(
        self,
        feed_cfg: str,
        user: str | None = None,
        model_id: str | None = None,
        priority: int = 5,
    ) -> str:
        """
        Send a scrape-feed request (similar to scheduler job).

        Args:
            feed_cfg: Path to the feed configuration file
            user: User name
            model_id: Model identifier
            priority: Request priority

        Returns:
            correlation_id for tracking the request
        """
        json_data = {"feed_cfg": feed_cfg}
        if user:
            json_data["user"] = user
        if model_id:
            json_data["model_id"] = model_id

        return await self.send_request("/scrape-feed", json_data, priority=priority)

    async def train_new_model(
        self,
        user: str,
        model_id: str,
        priority: int = 5,
    ) -> str:
        """
        Send a train-new-model request (similar to scheduler job).

        Args:
            user: User name
            model_id: Model identifier
            priority: Request priority

        Returns:
            correlation_id for tracking the request
        """
        json_data = {"user": user, "model_id": model_id}
        return await self.send_request("/train-new-model", json_data, priority=priority)

    async def regenerate(
        self,
        user: str,
        model_id: str,
        with_analysis: bool = False,
        since: str | None = None,
        priority: int = 5,
    ) -> str:
        """
        Send a regenerate request.

        Args:
            user: User name
            model_id: Model identifier
            with_analysis: Whether to include LLM analysis
            since: Start date for regeneration (ISO format)
            priority: Request priority

        Returns:
            correlation_id for tracking the request
        """
        json_data = {
            "user": user,
            "model_id": model_id,
            "with_analysis": with_analysis,
        }
        if since:
            json_data["since"] = since

        return await self.send_request("/regenerate", json_data, priority=priority)

    async def generate_report(
        self,
        user: str,
        model_id: str,
        reference_date: str | None = None,
        priority: int = 5,
    ) -> str:
        """
        Send a generate-report request.

        Args:
            user: User name
            model_id: Model identifier
            reference_date: Reference date for the report (ISO format)
            priority: Request priority

        Returns:
            correlation_id for tracking the request
        """
        json_data = {"user": user, "model_id": model_id}
        if reference_date:
            json_data["reference_date"] = reference_date

        return await self.send_request("/generate-report", json_data, priority=priority)

    async def wait_for_response(
        self, correlation_id: str, timeout: int = 120
    ) -> dict[str, Any] | None:
        """
        Wait for a response with a given correlation_id.

        Args:
            correlation_id: The correlation_id to wait for
            timeout: Timeout in seconds

        Returns:
            The response data or None if timeout
        """
        if correlation_id not in self.pending_requests:
            logger.warning(
                f"Correlation ID {correlation_id} not found in pending requests"
            )
            return None

        future = self.pending_requests[correlation_id]

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response: {correlation_id}")
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
            return None
        except Exception as e:
            logger.error(f"Error waiting for response {correlation_id}: {str(e)}")
            return None

    async def close(self):
        """Close connection"""
        if self._response_consumer_task:
            self._response_consumer_task.cancel()
            try:
                await self._response_consumer_task
            except asyncio.CancelledError:
                pass
        await self.queue_manager.close()


# Example usage and test simulation
async def main_test():
    parser = argparse.ArgumentParser(
        description="BERTrend Queue Client - Test requests"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/scrape-feed",
        help="Endpoint to call (e.g., /scrape-feed, /train-new-model)",
    )
    parser.add_argument("--user", type=str, default="test_user", help="User name")
    parser.add_argument("--model-id", type=str, default="test_model", help="Model ID")
    parser.add_argument(
        "--feed-cfg",
        type=str,
        default=None,
        help="Feed config path (for /scrape-feed)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Response timeout in seconds"
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Don't wait for response"
    )

    args = parser.parse_args()

    client = BertrendClient()
    await client.connect()

    try:
        # Send request based on endpoint
        if args.endpoint == "/scrape-feed":
            if not args.feed_cfg:
                logger.error("Error: --feed-cfg is required for /scrape-feed endpoint")
                exit(1)
            correlation_id = await client.scrape_feed(
                feed_cfg=args.feed_cfg,
                user=args.user,
                model_id=args.model_id,
            )
        elif args.endpoint == "/train-new-model":
            correlation_id = await client.train_new_model(
                user=args.user,
                model_id=args.model_id,
            )
        elif args.endpoint == "/regenerate":
            correlation_id = await client.regenerate(
                user=args.user,
                model_id=args.model_id,
            )
        elif args.endpoint == "/generate-report":
            correlation_id = await client.generate_report(
                user=args.user,
                model_id=args.model_id,
            )
        else:
            # Generic request
            correlation_id = await client.send_request(
                endpoint=args.endpoint,
                json_data={"user": args.user, "model_id": args.model_id},
            )

        logger.info(f"Request sent with correlation_id: {correlation_id}")

        if not args.no_wait:
            logger.info(f"Waiting for response (timeout: {args.timeout}s)...")
            response = await client.wait_for_response(
                correlation_id, timeout=args.timeout
            )

            if response:
                logger.info("\nResponse received:")
                logger.info(f"  Status: {response.get('status')}")
                if response.get("status") == "success":
                    logger.info(f"  Endpoint: {response.get('endpoint')}")
                    logger.info(
                        f"  Response: {json.dumps(response.get('response'), indent=2, default=str)}"
                    )
                else:
                    logger.error(f"  Error: {response.get('error')}")
            else:
                logger.error("Request timed out")
        else:
            logger.info("Request sent (not waiting for response)")

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    asyncio.run(main_test())
