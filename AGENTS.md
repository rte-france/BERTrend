### BERTrend Development Guidelines

This document provides project-specific information for developers working on BERTrend.

#### 1. Build/Configuration Instructions

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

#### 2. Testing Information

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

  @patch.dict(os.environ, {"BERTREND_BASE_DIR": "test_data"})
  def test_environment_setup():
      import bertrend
      assert bertrend.BASE_PATH == Path("test_data")
      assert bertrend.DATA_PATH.exists()

  def test_simple_logic():
      assert 1 + 1 == 2
  ```

#### 3. Project Structure

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

#### 4. Additional Development Information

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
