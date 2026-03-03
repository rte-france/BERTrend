# LLM Utilities API Reference

## Module: `bertrend/llm_utils/`

Utilities for LLM interactions, agent orchestration, newsletter generation, and prompt management.

---

## OpenAI Client (`openai_client.py`)

### Class: `APIType` (Enum)

Defines supported API backends:

- `APIType.OPENAI` — Direct OpenAI API.
- `APIType.LITELLM` — LiteLLM-compatible proxy.

### Class: `OpenAI_Client`

A unified client for interacting with OpenAI-compatible LLM APIs.

```python
OpenAI_Client(
    api_key: str = None,
    base_url: str = None,
    default_model: str = None,
)
```

Reads defaults from environment variables: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_DEFAULT_MODEL`.

#### Methods

| Method                                                                      | Description                                                                             |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `generate(prompt, model, temperature, max_tokens, **kwargs)`                | Generate a text completion from a single prompt string.                                 |
| `generate_from_history(messages, model, temperature, max_tokens, **kwargs)` | Generate a completion from a list of chat messages (`[{"role": ..., "content": ...}]`). |
| `parse(prompt, response_format, model, **kwargs)`                           | Generate a structured (parsed) response conforming to a Pydantic model schema.          |

---

## Agent Utilities (`agent_utils.py`)

Helpers for running OpenAI Agents SDK agents with concurrency and progress reporting.

### `run_runner_sync(runner_coro)`

Run an async `Runner` coroutine synchronously (convenience wrapper).

### Class: `BaseAgentFactory`

Abstract factory for creating configured agents.

```python
class BaseAgentFactory:
    def __init__(self, model: str = None)
    def _init_model(self) -> None
    def create_agent(self) -> Agent  # abstract
```

Subclass and implement `create_agent()` to define custom agents.

### Class: `ProcessingResult`

Container for a single agent processing result: `index`, `input_data`, `output`, `error`.

### Class: `AsyncAgentConcurrentProcessor`

Process a list of items through an agent concurrently with configurable parallelism.

```python
AsyncAgentConcurrentProcessor(
    agent_factory: BaseAgentFactory,
    max_concurrency: int = 5,
    chunk_size: int = 50,
)
```

#### Key Methods

| Method                                                  | Description                                                                                          |
|---------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `process_single_item(item, index, input_formatter)`     | Process one item through the agent.                                                                  |
| `process_list_concurrent(items, input_formatter, desc)` | Process all items with bounded concurrency and progress reporting. Returns `list[ProcessingResult]`. |

### `progress_reporter(total, desc)`

Async context manager that prints a live progress bar for concurrent processing.

---

## Newsletter Features (`newsletter_features.py`)

### `generate_newsletter(topics_df, language, ...)`

Generate a structured newsletter from topic analysis results using an LLM. Produces topic summaries, key insights, and
article references.

### `render_newsletter(newsletter, format)`

Render a `Newsletter` object into the specified format.

### `render_newsletter_html(newsletter)`

Render a newsletter as an HTML string.

### `render_newsletter_md(newsletter)`

Render a newsletter as a Markdown string.

---

## Newsletter Model (`newsletter_model.py`)

Pydantic models for structured newsletter data.

| Class        | Description                                                                    |
|--------------|--------------------------------------------------------------------------------|
| `Article`    | Represents a single article with `title`, `url`, `source`, `date`.             |
| `Topic`      | A newsletter topic section: `title`, `summary`, `articles`.                    |
| `Newsletter` | Top-level newsletter: `title`, `date`, `introduction`, `topics`, `conclusion`. |

---

## Prompts (`prompts.py`)

Contains prompt templates used by the LLM for:

- Topic summarization
- Newsletter section generation
- Signal interpretation

Prompts are defined as module-level string constants and support multi-language generation.
