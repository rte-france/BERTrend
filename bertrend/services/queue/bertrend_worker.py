#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
BERTrend Worker - Processes requests from RabbitMQ queue and calls FastAPI endpoints.

This worker receives HTTP-like requests from the queue (similar to what the scheduler
service sends) and processes them by calling the appropriate FastAPI router functions.
"""

import json
import traceback
from typing import Dict, Any

import pika
from loguru import logger

from bertrend.services.queue.queue_manager import QueueManager
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig

# Import router functions
from bertrend_apps.services.routers.data_provider import (
    scrape_api,
    auto_scrape_api,
    scrape_from_feed_api,
    generate_query_file_api,
)
from bertrend_apps.services.routers.bertrend_app import (
    train_new,
    regenerate,
    generate_report,
)

# Import request models
from bertrend_apps.services.models.data_provider_models import (
    ScrapeFeedRequest,
    ScrapeRequest,
    AutoScrapeRequest,
    GenerateQueryFileRequest,
)
from bertrend_apps.services.models.bertrend_app_models import (
    TrainNewModelRequest,
    RegenerateRequest,
    GenerateReportRequest,
)


# Mapping of endpoints to their handler functions and request models
ENDPOINT_HANDLERS = {
    "/scrape": (scrape_api, ScrapeRequest),
    "/auto-scrape": (auto_scrape_api, AutoScrapeRequest),
    "/scrape-feed": (scrape_from_feed_api, ScrapeFeedRequest),
    "/generate-query-file": (generate_query_file_api, GenerateQueryFileRequest),
    "/train-new-model": (train_new, TrainNewModelRequest),
    "/regenerate": (regenerate, RegenerateRequest),
    "/generate-report": (generate_report, GenerateReportRequest),
}


class BertrendWorker:
    """Worker that processes requests from RabbitMQ and calls FastAPI endpoints"""

    def __init__(self, config: RabbitMQConfig):
        self.config = config
        self.queue_manager = QueueManager(config)

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request by calling the appropriate FastAPI endpoint.

        Args:
            request_data: Dictionary containing:
                - endpoint: The API endpoint path (e.g., "/scrape-feed", "/train-new-model")
                - method: HTTP method (POST, GET, etc.) - currently only POST is supported
                - json_data: The request body data

        Returns:
            Dictionary containing the response or error information
        """
        try:
            endpoint = request_data.get("endpoint")
            method = request_data.get("method", "POST").upper()
            json_data = request_data.get("json_data", {})

            logger.info(f"Processing request: {method} {endpoint}")
            logger.debug(f"Request data: {json_data}")

            if endpoint not in ENDPOINT_HANDLERS:
                return {
                    "status": "error",
                    "error": f"Unknown endpoint: {endpoint}",
                    "available_endpoints": list(ENDPOINT_HANDLERS.keys()),
                }

            handler_func, request_model = ENDPOINT_HANDLERS[endpoint]

            # Create the request model from json_data
            try:
                request_obj = request_model(**json_data)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Invalid request data for {endpoint}: {str(e)}",
                }

            # Call the async handler function
            response = await handler_func(request_obj)

            # Convert response to dict if it's a Pydantic model
            if hasattr(response, "model_dump"):
                response_dict = response.model_dump()
            elif hasattr(response, "dict"):
                response_dict = response.dict()
            else:
                response_dict = response

            return {
                "status": "success",
                "endpoint": endpoint,
                "response": response_dict,
            }

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def callback(
        self,
        ch: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
    ):
        """Callback function for processing messages from the queue"""
        import asyncio

        correlation_id = properties.correlation_id

        try:
            logger.info(f"Received request: {correlation_id}")

            # Parse request
            request_data = json.loads(body.decode())
            logger.info(f"Request endpoint: {request_data.get('endpoint')}")

            # Process request (run async function in event loop)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response_data = loop.run_until_complete(
                    self.process_request(request_data)
                )
            finally:
                loop.close()

            # Add metadata
            response_data["correlation_id"] = correlation_id

            # Publish response if reply_to is specified
            if properties.reply_to:
                self.queue_manager.publish_response(
                    response_data=response_data, correlation_id=correlation_id
                )

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(
                f"Completed request: {correlation_id} - Status: {response_data.get('status')}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {str(e)}")
            # Reject and don't requeue invalid messages
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())

            # Requeue for retry (will go to DLQ after max retries)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start(self):
        """Start the worker"""
        logger.info("Starting BERTrend worker...")
        logger.info(f"Available endpoints: {list(ENDPOINT_HANDLERS.keys())}")

        try:
            self.queue_manager.connect()
            self.queue_manager.consume_requests(
                callback=self.callback, prefetch_count=self.config.prefetch_count
            )
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            self.queue_manager.close()


def main():
    """Main entry point"""
    config = RabbitMQConfig()
    worker = BertrendWorker(config)
    worker.start()


if __name__ == "__main__":
    main()
