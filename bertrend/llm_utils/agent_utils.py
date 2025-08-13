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

    input_index: int  # to keep track of link output/input
    output: str | None = None
    error: str | None = None
    processing_time: float = 0.0
    timeout: bool = False


class AsyncAgentConcurrentProcessor:
    """
    Async concurrent processor using OpenAI Agents SDK
    """

    def __init__(self, agent: Agent, max_concurrent: int = 5, timeout: float = 300.0):
        """
        Initialize the processor with agent configuration

        Args:
            agent: an OpenAI Agent
            max_concurrent: Maximum number of concurrent tasks
            timeout: Timeout for individual tasks in seconds (default: 5 minutes)
        """
        self.agent = agent
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_item(
        self, item: object, item_index: int = 0
    ) -> ProcessingResult:
        """
        Process a single item using the OpenAI Agent with timeout and proper error handling

        Args:
            item: Input data to process

        Returns:
            ProcessingResult containing the output or error
        """
        start_time = time.time()
        result = ProcessingResult(input_index=item_index)

        try:
            # Use asyncio.wait_for to add timeout protection
            async with asyncio.timeout(self.timeout):
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
                    except Exception as e:
                        result.error = f"Agent execution error: {str(e)}"
                        logger.error(f"Error processing item {str(item)[:50]}: {e}")

        except asyncio.TimeoutError:
            result.error = f"Task timed out after {self.timeout} seconds"
            result.timeout = True
            logger.warning(f"Item {str(item)[:50]} timed out after {self.timeout}s")
        except Exception as e:
            result.error = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error processing item {str(item)[:50]}: {e}")

        result.processing_time = time.time() - start_time
        return result

    async def process_list_concurrent(
        self,
        input_list: list[object],
        progress_callback: Optional[callable] = None,
        chunk_size: int | None = None,
        overall_timeout: float | None = None,
    ) -> list[ProcessingResult]:
        """
        Process a list of items concurrently using OpenAI Agents with improved error handling

        Args:
            input_list: List of items to process
            progress_callback: Optional callback for progress updates
            chunk_size: Optional chunk size for processing large lists in batches
            overall_timeout: Optional overall timeout for the entire operation

        Returns:
            List of ProcessingResult objects
        """
        total_items = len(input_list)

        if not input_list:
            logger.warning("Empty input list provided")
            return []

        if chunk_size and total_items > chunk_size:
            return await self._process_in_chunks(
                input_list, chunk_size, progress_callback, overall_timeout
            )

        logger.debug(
            f"Starting concurrent processing of {total_items} items with {self.max_concurrent} max concurrent, timeout: {self.timeout}s"
        )

        try:
            # Create tasks for all items
            tasks = [
                asyncio.create_task(self.process_single_item(item, i), name=f"item_{i}")
                for i, item in enumerate(input_list)
            ]

            # Process with overall timeout if specified
            if overall_timeout:
                async with asyncio.timeout(overall_timeout):
                    results = await self._collect_results_with_progress(
                        tasks, total_items, progress_callback
                    )
            else:
                results = await self._collect_results_with_progress(
                    tasks, total_items, progress_callback
                )

            sorted_results = sorted(
                results, key=lambda x: x.input_index
            )  # return list sorted by input index
            return sorted_results

        except asyncio.TimeoutError:
            logger.error(f"Overall operation timed out after {overall_timeout}s")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Return partial results with timeout errors for incomplete items
            results = []
            for i, task in enumerate(tasks):
                if task.done() and not task.cancelled():
                    try:
                        results.append(await task)
                    except Exception as e:
                        results.append(
                            ProcessingResult(
                                input_index=i,
                                error=f"Task failed: {str(e)}",
                            )
                        )
                else:
                    results.append(
                        ProcessingResult(
                            input_index=i,
                            error="Overall timeout exceeded",
                            timeout=True,
                        )
                    )
            sorted_results = sorted(
                results, key=lambda x: x.input_index
            )  # return list sorted by input index
            return sorted_results

        except Exception as e:
            logger.error(f"Unexpected error in process_list_concurrent: {e}")
            # Return error results for all items
            return [
                ProcessingResult(input_index=i, error=f"Processing failed: {str(e)}")
                for i, item in enumerate(input_list)
            ]

    async def _collect_results_with_progress(
        self,
        tasks: list[asyncio.Task],
        total_items: int,
        progress_callback: Optional[callable] = None,
    ) -> list[ProcessingResult]:
        """
        Collect results from tasks with progress reporting and proper error handling
        """
        results = [None] * len(tasks)  # Pre-allocate results list
        completed = 0

        # Use asyncio.as_completed with timeout protection
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                # Find the index of this task to maintain order
                task_name = coro.get_name() if hasattr(coro, "get_name") else None
                if task_name and task_name.startswith("item_"):
                    try:
                        idx = int(task_name.split("_")[1])
                        results[idx] = result
                    except (ValueError, IndexError):
                        # Fallback: append to first available slot
                        for i, slot in enumerate(results):
                            if slot is None:
                                results[i] = result
                                break
                else:
                    # Fallback: append to first available slot
                    for i, slot in enumerate(results):
                        if slot is None:
                            results[i] = result
                            break

                completed += 1

                if progress_callback:
                    progress_callback(completed, total_items, result)

                logger.debug(f"Completed {completed}/{total_items} tasks")

            except Exception as e:
                logger.error(f"Error collecting result: {e}")
                # Create error result for failed task
                error_result = ProcessingResult(
                    input_index=completed,
                    error=str(e),
                )
                results[completed] = error_result
                completed += 1

        # Filter out None values (shouldn't happen, but safety check)
        final_results = [r for r in results if r is not None]

        return final_results

    async def _process_in_chunks(
        self,
        input_list: list[object],
        chunk_size: int,
        progress_callback: Optional[callable] = None,
        overall_timeout: float | None = None,
    ) -> list[ProcessingResult]:
        """
        Process large lists in chunks to manage memory and rate limits
        """
        all_results = []
        total_items = len(input_list)
        processed_items = 0
        start_time = time.time()

        logger.info(f"Processing {total_items} items in chunks of {chunk_size}")

        for i in range(0, total_items, chunk_size):
            # Check overall timeout
            if overall_timeout and (time.time() - start_time) > overall_timeout:
                logger.warning("Overall timeout exceeded during chunked processing")
                # Add timeout results for remaining items
                remaining_items = input_list[i:]
                for item in remaining_items:
                    all_results.append(
                        ProcessingResult(
                            input_index=input_list.index(item),
                            error="Overall timeout exceeded",
                            timeout=True,
                        )
                    )
                break

            chunk = input_list[i : i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (total_items + chunk_size - 1) // chunk_size

            logger.info(
                f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} items)"
            )

            try:
                # Calculate remaining timeout for this chunk
                remaining_timeout = None
                if overall_timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(
                        15.0, overall_timeout - elapsed
                    )  # At least 15s per chunk

                # Process chunk
                chunk_results = await self.process_list_concurrent(
                    chunk,
                    progress_callback=None,  # We'll handle progress at the chunk level
                    overall_timeout=remaining_timeout,
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

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}: {e}")
                # Add error results for this chunk
                for item in chunk:
                    all_results.append(
                        ProcessingResult(
                            input_index=input_list.index(item),
                            error=f"Chunk processing failed: {str(e)}",
                        )
                    )
                processed_items += len(chunk)

        return all_results


def progress_reporter(current: int, total: int, result: ProcessingResult):
    """Enhanced progress callback function"""
    percentage = (current / total) * 10
    if result.timeout:
        status = "⏱"  # Timeout symbol
    elif result.output:
        status = "✓"  # Success
    else:
        status = "✗"  # Error

    logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) {status}")
    if result.error:
        logger.debug(f"Last error: {result.error[:100]}...")
