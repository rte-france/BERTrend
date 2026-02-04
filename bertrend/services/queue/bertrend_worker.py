#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
BERTrend Worker - Processes requests from RabbitMQ queue.

This worker receives HTTP-like requests from the queue and processes them
by calling the appropriate core logic functions.
"""

import asyncio
import traceback
from typing import Any

import aio_pika
import msgpack
import pandas as pd
from loguru import logger

from bertrend.services.queue.queue_manager import QueueManager
from bertrend.services.queue.rabbitmq_config import RabbitMQConfig
from bertrend_apps.newsletters.newsletter_generation import process_newsletter
from bertrend_apps.prospective_demo.automated_report_generation import (
    generate_automated_report,
)
from bertrend_apps.prospective_demo.process_new_data import (
    regenerate_models,
    train_new_model,
)
from bertrend_apps.data_provider.data_provider_utils import (
    auto_scrape,
    generate_query_file,
    scrape,
    scrape_feed_from_config,
)

# Import request models
from bertrend_apps.services.models.bertrend_app_models import (
    GenerateReportRequest,
    RegenerateRequest,
    TrainNewModelRequest,
)
from bertrend_apps.services.models.data_provider_models import (
    AutoScrapeRequest,
    GenerateQueryFileRequest,
    ScrapeFeedRequest,
    ScrapeRequest,
)
from bertrend_apps.services.models.newsletters_models import NewsletterRequest


# Wrapper functions to match the core logic with the request models
async def handle_scrape(req: ScrapeRequest):
    return await asyncio.to_thread(
        scrape,
        keywords=req.keywords,
        provider=req.provider,
        after=req.after,
        before=req.before,
        max_results=req.max_results,
        save_path=req.save_path,
        language=req.language,
    )


async def handle_auto_scrape(req: AutoScrapeRequest):
    return await asyncio.to_thread(
        auto_scrape,
        requests_file=req.requests_file,
        max_results=req.max_results,
        provider=req.provider,
        save_path=req.save_path,
        language=req.language,
        evaluate_articles_quality=req.evaluate_articles_quality,
        minimum_quality_level=req.minimum_quality_level,
    )


async def handle_scrape_feed(req: ScrapeFeedRequest):
    return await asyncio.to_thread(scrape_feed_from_config, req.feed_cfg)


async def handle_generate_query_file(req: GenerateQueryFileRequest):
    return await asyncio.to_thread(
        generate_query_file,
        keywords=req.keywords,
        after=req.after,
        before=req.before,
        interval=req.interval,
        save_path=req.save_path,
    )


async def handle_train_new(req: TrainNewModelRequest):
    return await asyncio.to_thread(
        train_new_model, model_id=req.model_id, user_name=req.user
    )


async def handle_regenerate(req: RegenerateRequest):
    return await asyncio.to_thread(
        regenerate_models,
        model_id=req.model_id,
        user=req.user,
        with_analysis=req.with_analysis,
        since=pd.Timestamp(req.since) if req.since else None,
    )


async def handle_generate_report(req: GenerateReportRequest):
    return await asyncio.to_thread(
        generate_automated_report,
        user=req.user,
        model_id=req.model_id,
        reference_date=req.reference_date,
    )


async def handle_generate_newsletters(req: NewsletterRequest):
    return await asyncio.to_thread(
        process_newsletter, req.newsletter_toml_path, req.data_feed_toml_path
    )


# Mapping of endpoints to their handler functions and request models
ENDPOINT_HANDLERS = {
    "/scrape": (handle_scrape, ScrapeRequest),
    "/auto-scrape": (handle_auto_scrape, AutoScrapeRequest),
    "/scrape-feed": (handle_scrape_feed, ScrapeFeedRequest),
    "/generate-query-file": (handle_generate_query_file, GenerateQueryFileRequest),
    "/train-new-model": (handle_train_new, TrainNewModelRequest),
    "/regenerate": (handle_regenerate, RegenerateRequest),
    "/generate-report": (handle_generate_report, GenerateReportRequest),
    "/generate-newsletters": (handle_generate_newsletters, NewsletterRequest),
}


class BertrendWorker:
    """Worker that processes requests from RabbitMQ"""

    def __init__(self, config: RabbitMQConfig):
        self.config = config
        self.queue_manager = QueueManager(config)

    async def process_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process a request by calling the appropriate core logic function.

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

    async def callback(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
    ):
        """Callback function for processing messages from the queue"""
        correlation_id = message.correlation_id

        try:
            logger.info(f"Received request: {correlation_id}")

            # Parse request
            request_data = msgpack.unpackb(message.body)
            logger.info(f"Request endpoint: {request_data.get('endpoint')}")

            # Process request
            response_data = await self.process_request(request_data)

            # Add metadata
            response_data["correlation_id"] = correlation_id

            # Publish response if reply_to is specified
            if message.reply_to:
                await self.queue_manager.publish_response(
                    response_data=response_data, correlation_id=correlation_id
                )

            # Acknowledge message
            await message.ack()
            logger.info(
                f"Completed request: {correlation_id} - Status: {response_data.get('status')}"
            )

        except (msgpack.UnpackException, msgpack.ExtraData) as e:
            logger.error(f"Invalid msgpack in message: {str(e)}")
            # Reject and don't requeue invalid messages
            await message.reject(requeue=False)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())

            # Requeue for retry (will go to DLQ after max retries)
            await message.nack(requeue=True)

    async def start(self):
        """Start the worker"""
        logger.info("Starting BERTrend worker...")
        logger.info(f"Available endpoints: {list(ENDPOINT_HANDLERS.keys())}")

        try:
            await self.queue_manager.connect()
            await self.queue_manager.consume_requests(
                callback=self.callback, prefetch_count=self.config.prefetch_count
            )

            # Wait until termination
            await asyncio.Future()

        except asyncio.CancelledError:
            logger.info("Worker interrupted")
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            await self.queue_manager.close()


def main():
    """Main entry point"""
    config = RabbitMQConfig()
    worker = BertrendWorker(config)
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
