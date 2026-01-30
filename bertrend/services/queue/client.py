#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
BERTrend Client - Sends requests to BERTrend service via RabbitMQ queue.

This client simulates requests similar to what the scheduler service sends,
allowing testing of the queue-based execution flow.
"""

import json
import time
from typing import Dict, Any, Optional

from loguru import logger

from bertrend.services.queue.queue_manager import QueueManager
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig


class BertrendClient:
    """Client for sending requests to BERTrend service via RabbitMQ"""

    def __init__(self, config: Optional[RabbitMQConfig] = None):
        self.config = config or RabbitMQConfig()
        self.queue_manager = QueueManager(self.config)
        self.queue_manager.connect()
        self.pending_requests: Dict[str, Any] = {}

    def send_request(
        self,
        endpoint: str,
        json_data: Dict[str, Any],
        method: str = "POST",
        priority: int = 5,
    ) -> str:
        """
        Send a request to the BERTrend service via the queue.

        This method simulates the HTTP requests that the scheduler service sends,
        but routes them through RabbitMQ instead.

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

        correlation_id = self.queue_manager.publish_request(
            request_data=request_data, priority=priority
        )

        self.pending_requests[correlation_id] = {
            "status": "pending",
            "timestamp": time.time(),
        }
        logger.info(f"Sent request to {endpoint} with correlation_id: {correlation_id}")

        return correlation_id

    def scrape_feed(
        self,
        feed_cfg: str,
        user: Optional[str] = None,
        model_id: Optional[str] = None,
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

        return self.send_request("/scrape-feed", json_data, priority=priority)

    def train_new_model(
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
        return self.send_request("/train-new-model", json_data, priority=priority)

    def regenerate(
        self,
        user: str,
        model_id: str,
        with_analysis: bool = False,
        since: Optional[str] = None,
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

        return self.send_request("/regenerate", json_data, priority=priority)

    def generate_report(
        self,
        user: str,
        model_id: str,
        reference_date: Optional[str] = None,
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

        return self.send_request("/generate-report", json_data, priority=priority)

    def get_response(self, correlation_id: str, timeout: int = 120) -> Optional[Dict]:
        """
        Get response for a request (blocking).

        Args:
            correlation_id: Request correlation ID
            timeout: Timeout in seconds

        Returns:
            Response dictionary or None if timeout
        """
        start_time = time.time()

        def response_callback(ch, method, properties, body):
            if properties.correlation_id == correlation_id:
                response = json.loads(body.decode())
                self.pending_requests[correlation_id] = response
                ch.basic_ack(delivery_tag=method.delivery_tag)
                ch.stop_consuming()

        # Start consuming responses
        self.queue_manager.channel.basic_consume(
            queue=self.config.response_queue,
            on_message_callback=response_callback,
            auto_ack=False,
        )

        # Wait for response with timeout
        while time.time() - start_time < timeout:
            self.queue_manager.connection.process_data_events(time_limit=1)

            if correlation_id in self.pending_requests:
                response = self.pending_requests[correlation_id]
                if response.get("status") != "pending":
                    return response

        logger.warning(f"Timeout waiting for response: {correlation_id}")
        return None

    def close(self):
        """Close connection"""
        self.queue_manager.close()


# Example usage and test simulation
if __name__ == "__main__":
    import argparse

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

    try:
        # Send request based on endpoint
        if args.endpoint == "/scrape-feed":
            if not args.feed_cfg:
                print("Error: --feed-cfg is required for /scrape-feed endpoint")
                exit(1)
            correlation_id = client.scrape_feed(
                feed_cfg=args.feed_cfg,
                user=args.user,
                model_id=args.model_id,
            )
        elif args.endpoint == "/train-new-model":
            correlation_id = client.train_new_model(
                user=args.user,
                model_id=args.model_id,
            )
        elif args.endpoint == "/regenerate":
            correlation_id = client.regenerate(
                user=args.user,
                model_id=args.model_id,
            )
        elif args.endpoint == "/generate-report":
            correlation_id = client.generate_report(
                user=args.user,
                model_id=args.model_id,
            )
        else:
            # Generic request
            correlation_id = client.send_request(
                endpoint=args.endpoint,
                json_data={"user": args.user, "model_id": args.model_id},
            )

        print(f"Request sent with correlation_id: {correlation_id}")

        if not args.no_wait:
            print(f"Waiting for response (timeout: {args.timeout}s)...")
            response = client.get_response(correlation_id, timeout=args.timeout)

            if response:
                print(f"\nResponse received:")
                print(f"  Status: {response.get('status')}")
                if response.get("status") == "success":
                    print(f"  Endpoint: {response.get('endpoint')}")
                    print(
                        f"  Response: {json.dumps(response.get('response'), indent=2, default=str)}"
                    )
                else:
                    print(f"  Error: {response.get('error')}")
            else:
                print("Request timed out")
        else:
            print("Request sent (not waiting for response)")

    finally:
        client.close()
