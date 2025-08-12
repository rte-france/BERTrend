#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

from agents import RunConfig, Agent, Runner, OpenAIChatCompletionsModel
from agents.extensions.models.litellm_model import LitellmModel
from loguru import logger
from openai import AsyncAzureOpenAI

from bertrend.llm_utils.openai_client import AZURE_API_VERSION

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
# Disable tracing
run_config = RunConfig(tracing_disabled=True)


class BaseAgentFactory:

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: str = API_KEY,
        base_url: str = BASE_URL,
    ):
        self.model_name = model_name or os.getenv("OPENAI_DEFAULT_MODEL_NAME")
        self.api_key = api_key
        if not api_key:
            logger.error(
                "WARNING: OPENAI_API_KEY environment variable not found. Please set it before using OpenAI services."
            )
            raise EnvironmentError(f"OPENAI_API_KEY environment variable not found.")
        self.base_url = base_url
        if self.base_url == "":  # check empty env var
            self.base_url = None
        self._init_model()

    def _init_model(self):
        if not self.base_url:
            # assume standard openai model
            self.model = self.model_name
        elif "azure.com" in self.base_url:
            self.model = OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=AsyncAzureOpenAI(
                    api_key=self.api_key,
                    api_version=AZURE_API_VERSION,
                    azure_endpoint=self.base_url,
                ),
            )
        else:
            # assume openAI compatible model
            self.model = LitellmModel(
                model=self.model_name, api_key=self.api_key, base_url=self.base_url
            )

    def create_agent(self, **kwargs) -> Agent:
        return Agent(model=self.model, **kwargs)


@dataclass
class ProcessingResult:
    """Result container for processed items"""

    input_data: object
    output: str | None = None
    error: str | None = None
    processing_time: float = 0.0


class AsyncAgentConcurrentProcessor:
    """
    Async concurrent processor using OpenAI Agents SDK
    """

    def __init__(self, agent: Agent, max_concurrent: int = 5):
        """
        Initialize the processor with agent configuration

        Args:
            agent: an OpenAI Agent
            max_concurrent: Maximum number of concurrent tasks
        """
        self.agent = agent
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_item(self, item: object) -> ProcessingResult:
        """
        Process a single item using the OpenAI Agent

        Args:
            item: Input data to process

        Returns:
            ProcessingResult containing the output or error
        """
        start_time = time.time()
        result = ProcessingResult(input_data=item)

        async with self.semaphore:  # Limit concurrent executions
            try:
                # Execute the agent task
                response = await Runner.run(
                    starting_agent=self.agent,
                    input=item,
                    run_config=run_config,
                )
                result.output = (
                    response.final_output
                    if hasattr(response, "final_output")
                    else str(response)
                )
                logger.debug(f"Successfully processed item: {str(item)[:10]}...")

            except Exception as e:
                result.error = str(e)
                logger.error(f"Error processing item {str(item)[:10]}: {e}")

        result.processing_time = time.time() - start_time
        return result

    async def process_list_concurrent(
        self,
        input_list: list[object],
        progress_callback: Optional[callable] = None,
        chunk_size: int | None = None,
    ) -> list[ProcessingResult]:
        """
        Process a list of items concurrently using OpenAI Agents

        Args:
            input_list: List of items to process
            progress_callback: Optional callback for progress updates
            chunk_size: Optional chunk size for processing large lists in batches

        Returns:
            List of ProcessingResult objects
        """
        total_items = len(input_list)

        if chunk_size and total_items > chunk_size:
            return await self._process_in_chunks(
                input_list, chunk_size, progress_callback
            )

        logger.debug(
            f"Starting concurrent processing of {total_items} items with {self.max_concurrent} max concurrent"
        )

        # Create tasks for all items
        tasks = [self.process_single_item(item) for item in input_list]

        # Process tasks concurrently and collect results
        results = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, total_items, result)

        # Handle exceptions in gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ProcessingResult(
                    input_data=input_list[i], error=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

            if progress_callback:
                progress_callback(i + 1, total_items, processed_results[-1])

        return processed_results

    async def _process_in_chunks(
        self,
        input_list: list[object],
        chunk_size: int,
        progress_callback: Optional[callable] = None,
    ) -> list[ProcessingResult]:
        """
        Process large lists in chunks to manage memory and rate limits
        """
        all_results = []
        total_items = len(input_list)
        processed_items = 0

        logger.info(f"Processing {total_items} items in chunks of {chunk_size}")

        for i in range(0, total_items, chunk_size):
            chunk = input_list[i : i + chunk_size]

            logger.info(f"Processing chunk {i//chunk_size + 1} ({len(chunk)} items)")

            # Process chunk
            chunk_results = await self.process_list_concurrent(
                chunk,
                progress_callback=None,  # We'll handle progress at the chunk level
            )

            all_results.extend(chunk_results)
            processed_items += len(chunk)

            # Call progress callback for the chunk
            if progress_callback:
                for result in chunk_results:
                    progress_callback(processed_items, total_items, result)

            # Add a small delay between chunks to respect rate limits
            if i + chunk_size < total_items:  # Don't sleep after the last chunk
                await asyncio.sleep(1)

        return all_results


def progress_reporter(current: int, total: int, result: ProcessingResult):
    """Progress callback function"""
    percentage = (current / total) * 100
    status = "✓" if result.output else "✗"
    logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) {status}")
