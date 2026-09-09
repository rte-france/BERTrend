# BERTrend Development Guidelines

## Git Workflow

- **Never push.** Do not run `git push` (nor `git push --force`, nor push tags) under any circumstance. Committing
  locally is fine; publishing is the maintainer's decision.
- **Never open a pull request.** Do not run `gh pr create` or otherwise open, edit, or merge a pull request. Leave the
  work on a local branch and report what is ready.
- **One single line per commit message.** A commit message is exactly one line — a short, imperative description of
  the change — followed by the human author's `Signed-off-by:` trailer. Nothing else: no body, no bullet list, no
  `Co-authored-by:` trailer.
- **Never mention the LLM.** No reference of any kind to the model, agent, or assistant that co-authored the code —
  not in the commit message, not in trailers, not in code comments, not in documentation.
- **Branch naming convention.** Create a branch for the change (never commit directly on `main`) and prefix its name
  with the type of work, followed by a short kebab-case description:
  - `feat/` — new feature (e.g. `feat/creation-tools`)
  - `fix/` — bug fix (e.g. `fix/per-unit-comparison`)
  - `refactor/` — restructuring with no behavior change
  - `docs/` — documentation only
  - `test/` — tests only
  - `chore/` — tooling, dependencies, CI
- Never commit real secrets in `.env`/`.env.*` files. When adding a new environment variable, document it (with a
  comment) in `.env.template` rather than only setting it locally.


## Build/Configuration Instructions

BERTrend uses `uv` for environment management and `python-dotenv` for configuration.

- **Environment Setup**:
    - Use `uv` to create a virtual environment and install dependencies:
      ```bash
      uv venv --python 3.13
      source .venv/bin/activate
      uv pip install -e .
      ```
- **Environment Variables**:
    - Configuration is managed via a `.env` file at the repository root.
    - Key variables:
        - `BERTREND_BASE_DIR`: Base directory for data, models, and logs. Defaults to `~/.bertrend`.
        - `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_DEFAULT_MODEL`: LLM configuration.
        - `EMBEDDING_SERVICE_URL`, `BERTREND_CLIENT_SECRET`: Embedding service configuration.
    - The project automatically loads the `.env` file upon importing the `bertrend` package (see
      `bertrend/__init__.py`).

- **Config Files**:
    - Default configurations for BERTopic, BERTrend, and Services are located in `bertrend/config/` as `.toml` files.

## Testing Information

- **Test Runner**: The project uses `pytest`.
- **Running Tests**:
    - To run tests, ensure your `.env` file is correctly configured or override necessary variables.
    - **Important**: Tests that import `bertrend` will attempt to create directories at `BERTREND_BASE_DIR`. Ensure the
      user has write permissions to this path. For local testing, you can override it:
      ```bash
      BERTREND_BASE_DIR=./test_data pytest
      ```
- **Adding New Tests**:
    - Place tests in the `bertrend/tests/` directory.
    - Mocking: Use `unittest.mock` to mock external services or environment variables if needed.

- **Test Example**:
  A simple test case to verify the environment setup and basic logic:
  ```python
  import pytest
  from pathlib import Path
  import os
  from unittest.mock import patch

      assert bertrend.BASE_PATH == Path("test_data")
      assert bertrend.DATA_PATH.exists()


  def test_simple_logic():
      assert 1 + 1 == 2
  ```

## Project Structure

The BERTrend project is organized into two main packages and several supporting directories:

- **`bertrend/`**: The core library.
    - `BERTrend.py`, `BERTopicModel.py`: Core logic for neural topic modeling and trend analysis.
    - `topic_analysis/`, `trend_analysis/`: Implementation of specific analysis methods and visualizations.
    - `metrics/`: TEMPTopic and other stability/volatility metrics.
    - `llm_utils/`: Utilities for LLM interactions, prompts, and newsletter generation.
    - `services/`: Core backend services (Embedding server, Scheduling, Summarization).
    - `demos/`: Streamlit demonstrators (Topic Analysis, Weak Signals).
    - `tests/`: Unit and integration tests for the core library.

- **`bertrend_apps/`**: High-level applications and integration services.
    - `data_provider/`: Adapters for various data sources (RSS, Atom, ArXiv, Google News, etc.).
    - `newsletters/`: Automated newsletter generation logic.
    - `services/`: FastAPI-based services for data provision and app management.
    - `prospective_demo/`: A comprehensive "Prospective Demo" application.
    - `exploration/`: Miscellaneous tools for data visualization and geolocalization.

- **Other Directories**:
    - `data/`: Default location for local datasets (if configured).
    - `docs/`: Technical documentation and design plans.
    - `getting_started/`: Jupyter notebooks and guides for new users.

## 4. Additional Development Information

- **Code Style**:
    - Follow the existing style: `ruff` is used for linting and formatting (see `pyproject.toml`).
    - Indentation: 4 spaces.
- **Logging**:
    - The project uses `loguru` for logging.
    - Logs are automatically colorized and formatted.
    - The log path is determined by `BERTREND_LOG_PATH` (under `BERTREND_BASE_DIR`).
- **GPU Usage**:
    - The project attempts to find the best CUDA device automatically (see `BEST_CUDA_DEVICE` in
      `bertrend/__init__.py`).
    - You can manually set `CUDA_VISIBLE_DEVICES` in your `.env` file.
- **Dependencies**:
    - Some dependencies have specific version requirements to ensure stability. Refer to `pyproject.toml` for details.
